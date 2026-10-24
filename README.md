# Titanic и House Prices

Проект по классификации и регрессии. Для Titanic предсказывается выживание пассажира, для House Prices — стоимость дома.

Код находится в `src/`, настройки — в `configs/`, анализ данных — в `notebooks/`. Все команды запускаются через `main.py`.

## Результаты

| Задача | Модель | Средняя метрика CV | OOF |
|---|---|---:|---:|
| Titanic | HistGradientBoosting | accuracy 0.846224 | 0.846240 |
| House Prices | CatBoost, learning_rate=0.1 | log-RMSE 0.127212 | 0.128564 |

Сравнивались линейные модели, KNN, деревья, RandomForest, CatBoost, LightGBM, XGBoost, MLP и ансамбли. Все 111 результатов с параметрами сохранены в [таблице экспериментов](docs/experiments.csv).

Используются пять фиксированных фолдов: StratifiedKFold для Titanic и KFold для House Prices. Обработка признаков обучается внутри каждого фолда. В регрессии модель учится на ln(price), а predict возвращает цену. log-RMSE считается как корень из среднего квадрата разницы логарифмов цен.

## Установка

Нужен Python 3.12. Команды для PowerShell, из папки проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Данные

Скачать файлы соревнований [Titanic](https://www.kaggle.com/competitions/titanic/data) и [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) и разложить так:

```text
data/classification/
    train.csv
    test.csv
    gender_submission.csv
data/regression/
    train.csv
    test.csv
    sample_submission.csv
    data_description.txt
```

Данные, виртуальное окружение, модели и submission не входят в Git.

## Запуск

Обучение выбранных моделей и создание файлов для Kaggle:

```powershell
.\.venv\Scripts\python.exe main.py --config configs/final_classification.yaml --mode train --run-name titanic_01 --make-submission
.\.venv\Scripts\python.exe main.py --config configs/final_regression.yaml --mode train --run-name housing_01 --make-submission
```

В терминале выводятся метрики по фолдам, среднее, стандартное отклонение и OOF. Модели сохраняются в `artifacts/`, журнал — в `results/`, CSV — в `submissions/`:

- `classification_titanic_01.csv`
- `regression_housing_01.csv`

Для повторного запуска нужно другое имя в `--run-name`: существующие результаты не перезаписываются.

Прогноз сохранённой моделью без нового обучения:

```powershell
.\.venv\Scripts\python.exe main.py --mode inference --run-dir artifacts/classification/titanic_01 --submission-path submissions/titanic_predict.csv --trust-models
```

Нужен весь каталог run, а не один joblib. Загружать следует только собственные модели: joblib может выполнять код. Для готовых моделей также требуются Python 3.12.10 и те же версии библиотек.

Остальные режимы: `check` проверяет данные без обучения, `debug` обучает небольшую выборку, `ensemble` усредняет минимум два готовых run одной задачи. Например:

```powershell
.\.venv\Scripts\python.exe main.py --mode ensemble --run-dirs artifacts/classification/rf_20261018 artifacts/classification/hgb_20261018 --run-name average_01 --trust-models
```

Эти два run должны быть уже обучены. Для просмотра аргументов: `python main.py --help`.

## Тесты

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

17 тестов на искусственных данных. Файлы Kaggle для них не нужны.
