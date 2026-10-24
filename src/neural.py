"""CPU MLP: обучение батчами и остановка по внутренней validation."""

from copy import deepcopy
import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingNetwork(nn.Module):
    """Числа + обучаемые векторы категорий; индекс 0 зарезервирован под UNK."""

    def __init__(self, numeric_features, cardinalities, layers):
        super().__init__()
        self.numeric_features = numeric_features
        self.embeddings = nn.ModuleList([
            nn.Embedding(count, min(8, max(2, (count + 1) // 2))) for count in cardinalities
        ])
        self.layers = layers

    def forward(self, X):
        parts = [X[:, :self.numeric_features]]
        for column, embedding in enumerate(self.embeddings, start=self.numeric_features):
            parts.append(embedding((X[:, column].long() + 1)))
        return self.layers(torch.cat(parts, dim=1))


class TorchMLP(BaseEstimator):
    """Sklearn-совместимая оболочка. fit требует отдельную stopping-выборку.

    В регрессии обучаем стандартизованный ln(price); predict возвращает цену.
    После fit network_ содержит BEST веса, last_checkpoint_ — последнее состояние.
    CPU/seed/число потоков задаются явно; fit меняет RNG состояния процесса.
    """

    def __init__(self, hidden_sizes=(64, 32), dropout=0.1, learning_rate=0.001,
                 weight_decay=0.0001, batch_size=64, max_epochs=150,
                 patience=15, min_delta=0.0001, random_state=42,
                 batch_norm=False, activation="relu", optimizer="adamw",
                 scheduler="none", loss="default", embedding=False, numeric_features=0):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.batch_norm = batch_norm
        self.activation = activation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.embedding = embedding
        self.numeric_features = numeric_features

    def _array(self, X):
        result = np.asarray(X, dtype=np.float32)
        if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
            raise ValueError("MLP expects a non-empty finite 2D numeric array")
        return result

    def _target(self, y, rows):
        result = np.asarray(y, dtype=np.float64)
        if result.shape != (rows,) or not np.isfinite(result).all():
            raise ValueError("Target must be finite and match X rows")
        if self.task == "classification":
            if not np.isin(result, [0, 1]).all():
                raise ValueError("Classifier target must contain only 0/1")
        else:
            if (result <= 0).any():
                raise ValueError("Regression prices must be positive")
            result = np.log(result)
        return result

    def fit(self, X, y, *, validation_data):
        """Обучить только inner-fit; выбрать BEST по отдельной stopping-части."""
        if any(type(value) is not int or value < 1 for value in
               (self.batch_size, self.max_epochs, self.patience)):
            raise ValueError("batch_size/max_epochs/patience must be positive integers")
        if (not self.hidden_sizes or any(type(size) is not int or size < 1 for size in self.hidden_sizes)
                or not 0 <= self.dropout < 1 or not np.isfinite(self.learning_rate)
                or self.learning_rate <= 0 or not np.isfinite(self.weight_decay)
                or self.weight_decay < 0 or not np.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError("Invalid MLP architecture or optimizer settings")
        if (self.activation not in ("relu", "gelu") or self.optimizer not in ("adam", "adamw")
                or self.scheduler not in ("none", "cosine")
                or self.loss not in (("default", "weighted_bce") if self.task == "classification"
                                     else ("default", "huber"))):
            raise ValueError("Unknown activation/optimizer/scheduler/loss")
        X, X_valid = self._array(X), self._array(validation_data[0])
        if X.shape[1] != X_valid.shape[1]:
            raise ValueError("Train/stopping feature counts must match")
        y, y_valid = self._target(y, len(X)), self._target(validation_data[1], len(X_valid))
        if self.task == "classification" and not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("MLP training requires both classes")

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(2)
        self.n_features_in_ = X.shape[1]
        self.target_mean_ = float(y.mean()) if self.task == "regression" else 0.
        self.target_scale_ = max(float(y.std()), 1e-8) if self.task == "regression" else 1.
        y = ((y - self.target_mean_) / self.target_scale_).astype(np.float32)
        y_valid = ((y_valid - self.target_mean_) / self.target_scale_).astype(np.float32)

        layers, width = [], X.shape[1]
        cardinalities = []
        if self.embedding:
            if not 0 <= self.numeric_features < X.shape[1]:
                raise ValueError("Embedding needs at least one categorical column")
            categories = X[:, self.numeric_features:]
            if (categories < 0).any() or not np.equal(categories, categories.astype(int)).all():
                raise ValueError("Fit embedding input must contain non-negative categorical indices")
            cardinalities = (categories.max(axis=0).astype(int) + 2).tolist()
            width = self.numeric_features + sum(min(8, max(2, (count + 1) // 2)) for count in cardinalities)
        self.category_cardinalities_ = cardinalities
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(width, hidden))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.extend([nn.ReLU() if self.activation == "relu" else nn.GELU(), nn.Dropout(self.dropout)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        network = nn.Sequential(*layers)
        self.network_ = (EmbeddingNetwork(self.numeric_features, cardinalities, network)
                         if self.embedding else network)
        optimizer_class = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        optimizer = optimizer_class(self.network_.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
                     if self.scheduler == "cosine" else None)
        self.positive_weight_ = float((y == 0).sum() / (y == 1).sum()) if self.task == "classification" else None
        if self.task == "classification":
            weight = torch.tensor(self.positive_weight_) if self.loss == "weighted_bce" else None
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            criterion = nn.HuberLoss() if self.loss == "huber" else nn.MSELoss()
        generator = torch.Generator().manual_seed(self.random_state)
        # drop_last применяется только на обучении и только при singleton + BN.
        # Stopping/outer-validation/test всегда прогнозируются полностью.
        if self.batch_norm and (self.batch_size < 2 or len(X) < 2):
            raise ValueError("BatchNorm requires batch_size and training rows >=2")
        drop_singleton = self.batch_norm and len(X) % self.batch_size == 1
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                                  batch_size=self.batch_size, shuffle=True, generator=generator,
                                  drop_last=drop_singleton)
        valid_loader = DataLoader(TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid)),
                                  batch_size=self.batch_size, shuffle=False)
        self.history_ = []
        best_loss, stale = float("inf"), 0
        for epoch in range(1, self.max_epochs + 1):
            self.network_.train()
            total_train, seen = 0., 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.network_(batch_X).squeeze(-1), batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite train loss")
                loss.backward()
                optimizer.step()
                total_train += loss.item() * len(batch_y)
                seen += len(batch_y)

            self.network_.eval()
            total_valid = 0.
            with torch.inference_mode():
                for batch_X, batch_y in valid_loader:
                    loss = criterion(self.network_(batch_X).squeeze(-1), batch_y)
                    total_valid += loss.item() * len(batch_y)
            valid_loss = total_valid / len(y_valid)
            if not np.isfinite(valid_loss):
                raise ValueError("Non-finite validation loss")
            self.history_.append({"epoch": epoch, "train_loss": total_train / seen,
                                  "stopping_loss": valid_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
            if scheduler is not None:
                scheduler.step()
            checkpoint = {
                "epoch": epoch, "task": self.task, "params": self.get_params(),
                "input_dim": self.n_features_in_, "target_mean": self.target_mean_,
                "target_scale": self.target_scale_,
                "model_state_dict": deepcopy(self.network_.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": generator.get_state(),
                "history": deepcopy(self.history_), "stopping_loss": valid_loss,
                "scheduler_state_dict": deepcopy(scheduler.state_dict()) if scheduler else None,
                "category_cardinalities": cardinalities,
                "positive_weight": self.positive_weight_,
            }
            # min_delta определяет терпение; BEST — абсолютный минимум loss.
            if epoch == 1 or valid_loss < self.best_checkpoint_["stopping_loss"]:
                self.best_checkpoint_ = deepcopy(checkpoint)
            if valid_loss < best_loss - self.min_delta:
                best_loss, stale = valid_loss, 0
            else:
                stale += 1
            self.last_checkpoint_ = checkpoint
            if stale >= self.patience:
                break
        self.best_epoch_ = self.best_checkpoint_["epoch"]
        self.last_epoch_ = epoch
        self.network_.load_state_dict(self.best_checkpoint_["model_state_dict"])
        self.network_.eval()
        if self.task == "classification":
            self.classes_ = np.array([0, 1])
        return self

    def _output(self, X):
        check_is_fitted(self, "best_epoch_")
        X = self._array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted MLP")
        self.network_.eval()
        with torch.inference_mode():
            output = self.network_(torch.from_numpy(X)).squeeze(-1)
            if self.task == "classification":
                output = torch.sigmoid(output)
            result = output.numpy().astype(float)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite MLP output")
        return result


class TorchClassifier(ClassifierMixin, TorchMLP):
    """Бинарная MLP: sigmoid и столбцы вероятностей классов 0/1."""
    task = "classification"

    def predict_proba(self, X):
        probability = self._output(X)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self._output(X) >= .5).astype(int)


class TorchRegressor(RegressorMixin, TorchMLP):
    """MLP для стандартизованной ln(price), predict возвращает исходную цену."""
    task = "regression"

    def predict(self, X):
        result = np.exp(self._output(X) * self.target_scale_ + self.target_mean_)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("MLP predicted non-finite/non-positive prices")
        return result


def fit_torch_fold(pipeline, X, y, random_state):
    """Inner split только outer-train: outer-validation не выбирает эпоху.

    Imputer/encoder/scaler обучаем лишь на inner-fit, не на stopping-части.
    После early stopping не дообучаем на stopping-части ради простоты.
    """
    estimator = pipeline.named_steps["model"]
    fit_pos, stop_pos = train_test_split(
        np.arange(len(X)), test_size=.2, random_state=random_state,
        stratify=y if estimator.task == "classification" else None,
    )
    preprocessing = pipeline[:-1]
    fit_X = preprocessing.fit_transform(X.iloc[fit_pos], y.iloc[fit_pos])
    stop_X = preprocessing.transform(X.iloc[stop_pos])
    estimator.fit(fit_X, y.iloc[fit_pos], validation_data=(stop_X, y.iloc[stop_pos]))
    estimator.split_ = {"fit_index": X.index[fit_pos].tolist(),
                        "stopping_index": X.index[stop_pos].tolist()}
    return pipeline
