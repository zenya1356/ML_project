# Final ML Project — скорректированный план

Обновлено: 5 октября 2026 года.

План уточнён по заметкам «Пайплайн классического ML» и «Пайплайн DL».

## Как мы работаем

- Пользователь самостоятельно пишет код, создаёт файлы и запускает команды.
- Ассистент объясняет один небольшой шаг, ожидаемый результат и проверяет вывод.
- Не переходим дальше, пока текущий шаг не работает.
- Любая новая обработка, модель или feature engineering принимается только после честного сравнения на cross-validation.

## Цель и архитектура

Для Titanic и House Prices строится единый воспроизводимый табличный пайплайн:

```text
Config
  → загрузка и проверка данных
  → EDA и гипотезы
  → фиксированный split
  → fold-safe preprocessing и feature engineering
  → обучение по фолдам
  → OOF, метрики и артефакты
  → inference без повторного обучения
  → submission и ансамбль
```

Обязанности разделяются так:

- `configs/` — параметры задачи и эксперимента;
- `notebooks/` — EDA, графики и черновые гипотезы;
- `src/data.py` — загрузка и базовая проверка таблиц;
- `src/preprocessing.py` — обучаемые преобразования с контрактом `fit/transform`;
- `src/features.py` — генерация признаков;
- `src/validation.py` — split, folds, метрики и OOF;
- `src/models.py` — создание моделей;
- `src/train.py` — обучение по фолдам и сохранение результатов;
- `src/utils.py` — seed, конфиги, логи и служебные функции;
- `main.py` — orchestration, а не вся логика проекта;
- `artifacts/` — сохранённые preprocessors, модели, folds, OOF, конфиги и логи;
- `results/` — таблица экспериментов;
- `submissions/` — готовые файлы для Kaggle.

Новые модули и каталог `artifacts/` создаются только тогда, когда до них дойдёт план.

## Этап 0. Подготовка и минимальный запуск

### Уже выполнено

- структура репозитория и `.venv`;
- базовые зависимости;
- данные двух соревнований;
- targets, ID, метрики и форматы submission;
- начальные YAML-конфиги.

### Осталось

1. Написать `load_data` в `src/data.py` и проверить Titanic.
2. Добавить `split_features_target` и проверить обе задачи.
3. Научить `main.py` принимать `--config`.
4. Загрузить YAML через `yaml.safe_load`.
5. Сделать минимальный запуск обеих задач одной командой.
6. Добавить понятные проверки схемы: target только в train, ID в train/test, одинаковый состав признаков.
7. Сделать первый Git-коммит.

На этом этапе `main.py` только читает конфиг и данные. Он ещё ничего не обучает.

## Этап 1. EDA и постановка гипотез

Создать два notebook:

- `notebooks/classification_eda.ipynb`;
- `notebooks/regression_eda.ipynb`.

Проверить shape, типы, уникальные значения, дубликаты, пропуски, target, числовые и категориальные признаки, выбросы, константы, ID-like поля, корреляции и возможный leakage.

Для House Prices отдельно различать:

- настоящий неизвестный пропуск;
- категорию «объект отсутствует», которая записана как `NA`.

После каждого блока записывать текстовый вывод и формулировать проверяемую гипотезу. Notebook остаётся исследовательским; только подтверждённые идеи переносятся в `src/`.

## Этап 2. Валидация до preprocessing и моделей

1. Classification: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. Regression: `KFold(n_splits=5, shuffle=True, random_state=42)`.
3. Зафиксировать fold для каждой строки, чтобы все эксперименты сравнивались на одном split.
4. Проверить, нет ли групповой или временной зависимости, требующей другого split.
5. Реализовать сохранение:
   - score каждого фолда;
   - CV mean и std;
   - OOF-предсказаний;
   - test-предсказаний;
   - времени обучения.

OOF — предсказание для объекта от модели, которая не обучалась на этом объекте. OOF используется для анализа ошибок, выбора ансамбля и stacking.

## Этап 3. Baseline и первый полный эксперимент

Сначала пройти весь путь в debug-режиме на небольшой выборке, затем выполнить полный запуск.

Baseline:

- classification: DummyClassifier, LogisticRegression;
- regression: DummyRegressor, LinearRegression, Ridge.

Preprocessing оформлять через `Pipeline`/`ColumnTransformer`, чтобы imputer, encoder и scaler обучались только на train-части очередного фолда.

Каждый эксперимент сохраняет:

- имя запуска;
- копию конфига;
- seed и split;
- параметры модели;
- feature set;
- preprocessing;
- fold scores, mean/std и время;
- OOF и test predictions;
- итоговую строку в таблице результатов.

После этого должен работать первый reproducible train-run и первый submission.

## Этап 4. Preprocessing-эксперименты

Проверять по одному изменению и сравнивать с baseline:

1. обработка пропусков;
2. категориальное кодирование;
3. StandardScaler и RobustScaler;
4. константные признаки;
5. коррелированные признаки;
6. выбросы и transformations.

Правило: `fit` выполняется на train-фолде, `transform` — на validation и test. Нельзя заново определять категории, удаляемые колонки или статистики на inference.

Сохранять состояние preprocessing:

- обученный imputer/scaler/encoder;
- список удалённых колонок;
- категориальные колонки и mapping;
- окончательный состав и порядок признаков.

## Этап 5. Classic ML и boosting

Classification:

- LogisticRegression;
- KNN;
- DecisionTreeClassifier;
- RandomForestClassifier.

Regression:

- LinearRegression, Ridge, Lasso, ElasticNet;
- KNNRegressor;
- DecisionTreeRegressor;
- RandomForestRegressor.

После baseline установить и проверить:

- CatBoost;
- LightGBM;
- XGBoost.

Обучение организовать единым циклом по моделям и фолдам. Абстракцию `Solver`/trainer вводить после того, как один baseline полностью прошёл цикл, чтобы не строить преждевременно сложную архитектуру.

Сохранять модель каждого фолда и связанное с ней состояние preprocessing. Финальное inference не должно переобучать модель.

## Этап 6. Feature engineering

Проверять гипотезы по одной:

- декомпозиция дат;
- interaction features;
- отношения и суммы;
- `log1p`/`sqrt` для обоснованно асимметричных признаков;
- предметные признаки Titanic и House Prices.

Агрегаты, target encoding и другие статистические признаки вычислять внутри фолда. Для train-объектов при необходимости использовать cross-fitting. Нельзя использовать validation-target при построении признаков.

Кэш подготовленных данных добавлять только при реально дорогой обработке. Кэш должен зависеть от версии данных, feature pipeline и split.

## Этап 7. DNN как отдельное семейство моделей

PyTorch устанавливается только на этом этапе.

Минимальная последовательность:

1. табличный `Dataset` и `DataLoader`;
2. простой MLP;
3. корректные train/validation loops;
4. OOF по тем же folds;
5. early stopping;
6. best checkpoint для inference;
7. last checkpoint для resume;
8. сохранение метрик и логов.

Обязательно:

- фиксировать Python/NumPy/PyTorch seed;
- использовать `model.train()` на train;
- использовать `model.eval()` и `torch.no_grad()`/`inference_mode()` на validation;
- train DataLoader может перемешивать данные, validation — нет;
- оценивать все validation-объекты;
- сначала прогонять debug-режим.

Архитектурные эксперименты:

- basic MLP;
- дополнительные hidden layers;
- BatchNorm;
- Dropout;
- разные размеры слоёв и activations;
- Adam/AdamW;
- scheduler.

AMP, gradient accumulation, clipping, warmup и сложный resume включать только при необходимости. Они не являются обязательными для первой работающей DNN.

## Этап 8. Inference, ensembles и submissions

Добавить отдельный режим запуска:

- `train` — обучает, оценивает и сохраняет артефакты;
- `inference` — загружает готовые преобразования и модели и создаёт predictions;
- `debug` — проходит тот же путь на небольшом объёме данных.

На inference проверять точный состав и порядок признаков.

Сравнить по OOF:

- лучшую отдельную модель;
- averaging;
- soft voting для classification;
- stacking.

Веса и состав ансамбля выбирать по validation/OOF, а не по test или случайным leaderboard-попыткам. Для stacking использовать OOF-предсказания базовых моделей.

## Этап 9. Финализация

Проверить, что из чистого окружения достаточно:

```powershell
python -m pip install -r requirements.txt
python main.py --config configs/classification.yaml --mode train
python main.py --config configs/classification.yaml --mode inference
```

Аналогично для regression.

Финальный набор артефактов:

- configs;
- зафиксированные folds;
- preprocessing/data checkpoints;
- модели по фолдам;
- best/last DNN checkpoints;
- OOF и test predictions;
- таблица экспериментов;
- логи;
- submissions;
- README с выводами и инструкциями запуска.

## Приоритеты

### P0 — обязательно

- корректный split и отсутствие leakage;
- baseline и единый CV;
- OOF;
- воспроизводимый запуск одной командой;
- сохранение preprocessing, моделей и схемы признаков;
- отдельный inference без переобучения;
- минимум два boosting-подхода;
- DNN для обеих задач;
- таблица экспериментов и README.

### P1 — желательно

- debug-режим;
- все три boosting-библиотеки;
- feature importance;
- feature engineering;
- averaging и stacking;
- best/last/resume для DNN;
- кэш дорогих признаков.

### P2 — только при наличии времени и гипотезы

- nested CV;
- псевдоразметка;
- embeddings категорий;
- сложный tuning;
- AMP, accumulation и warmup;
- оптимизация памяти и downcast;
- сложные многоступенчатые ансамбли.

## Оценка времени и текущий прогресс

Исходные 15 часов достаточны только для компактного учебного прототипа. Для полного воспроизводимого варианта с двумя задачами, OOF, сохранением артефактов, DNN и ансамблями реалистичнее планировать 25–35 часов.

Текущий прогресс:

- подготовительный этап — около 75%;
- полный скорректированный план — около 3–4%.

## Точная следующая точка

Текущий шаг не меняется: пользователь самостоятельно пишет `load_data` в `src/data.py`, проверяет загрузку Titanic, затем добавляет `split_features_target` и проверяет обе задачи.

