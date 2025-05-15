# Iris-MLflow 

Минимальный, но полностью рабочий пример MLOps-контура  
**PostgreSQL + MinIO + MLflow 2.22** (+ GitHub Actions).


## Что это делает

* поднимает MLflow-сервер в Docker-Compose;  
* хранит метаданные в Postgres, артефакты — в MinIO-bucket `mlflow`;  
* обучает `RandomForestClassifier` на датасете Iris и логирует всё в MLflow;  
* при каждом push в GitHub workflow воспроизводит те же действия и кладёт новую версию модели в Model Registry.

## Prerequisites

| Софт | Проверенная версия |
|------|-------------------|
| Docker Desktop | ≥ 4.31 (multi-arch) |
| Docker Compose V2 | идёт в составе Desktop |
| Python | 3.10 + (только если хотите локально запускать `train.py`) |

## Структура репозитория

```
.
├── docker-compose.yaml
├── Dockerfile.mlflow          # кастомный образ MLflow + psycopg2
├── src/
│   ├── train.py
│   └── requirements.txt
├── tests/
│   └── test_inference.py
├── .github/workflows/ci.yml
└── README.md                 
```

## Быстрый старт локально

```bash
# 1. клонируем
git clone https://github.com/buliginkirill/iris-mlflow.git
cd iris-mlflow

# 2. поднимаем стек
docker compose up -d        # db + minio + mlflow (+ init-bucket)

# 3. смотрим UI
open http://localhost:5050   # MLflow
open http://localhost:9001   # MinIO (login: mlflow / mlflow123)

# 4. обучаем модель
export MLFLOW_TRACKING_URI=http://localhost:5050
python src/train.py --n_estimators 150 --max_depth 7
```

## Почему использование MLFlow помогает именно в нашей реализации 

В нашем демонстрационном проекте мы строим весь ML-конвейер вокруг MLflow, потому что именно он превращает разрозненный набор скриптов в воспроизводимую систему. При каждом запуске train.py MLflow автоматически фиксирует параметры Random Forest’а, метрики на обучающей и тестовой выборках, версию кода и точную «замороженную» среду Python. Эти данные сохраняются в Postgres, а тяжёлые артефакты — сериализованная модель, графики, окружение — кладутся в MinIO-бакет, так что репозиторий остаётся лёгким, а модели доступны по S3-URL-ам. Через встроенный UI мы сразу видим, какой набор гиперпараметров дал лучшую точность, можем сравнить ранние эксперименты и при необходимости воспроизвести любой запуск даже через месяцы. Model Registry избавляет от хаоса с файлами «model_final.pickle» — лучшая версия регистрируется под именем iris_rf, получает понятную метку (v1, v2, v3) и продвигается по стадиям Staging/Production прямо из CI. GitHub Actions используют тот же REST-endpoint, что и локальная машина, поэтому pipeline в облаке ведёт себя идентично локальному запуску: поднимает временный MLflow-сервер, прогоняет обучение, логирует результат и, при успехе, публикует новую версию модели. Таким образом MLflow даёт прозрачность экспериментов, единый формат хранения артефактов и простое версионирование моделей — без него проект свёлся бы к набору не-отслеживаемых скриптов, а с ним превращается в полноценную, легко поддерживаемую MLOps-схему.