# Antibody Safety & Property Scoring Pipeline (SOTA‑модели)

Этот проект реализует **единый веб‑pipeline** для оценки аминокислотной последовательности (в том числе антител) по 5 группам свойств:

1. Токсичность  
2. Аллергенность  
3. Иммуногенность (B‑ и T‑клеточные эпитопы)  
4. Стабильность белка / антитела  
5. Стандартные физико‑химические свойства белка (ProtParam‑стиль)

API построено на **FastAPI**, а каждая из scoring‑моделей (токсичность, аллергенность, B/T‑эпитопы, стабильность) **регистрируется и загружается из MLflow Model Registry**, обеспечивая воспроизводимость, версионирование и удобную интеграцию с MLOps‑инструментами.

---

## 🧬 Цель проекта

- Обеспечить **интерактивный веб‑демо‑стенд** для заказчика: ввод последовательности → JSON‑ответ с подробным скорингом по каждой группе свойств.
- Дать **ML‑инженерам** шаблон:
  - обучения и регистрации SOTA‑моделей (StrucToxNet‑подобная, AllergenAI / pLM4Alg, IEDB‑style B/T‑эпитопы, Stability Oracle‑style стабильность),
  - деплоя через FastAPI + MLflow в Docker / на хостинг.

---

## 📚 Список SOTA‑моделей в пайплайне

В проекте используются следующие типы моделей (реализованы как `mlflow.pyfunc`‑модели):

### 1. Токсичность
- **SOTA‑подход**: `StrucToxNet` / PLM‑based классификатор на основе протеин‑языковой модели (ProtT5‑подобная), обученный на токсичных/нетоксичных пептидах.
- **Фичи при обучении**:
  - PLM‑эмбеддинги последовательности,
  - композиция аминокислот,
  - частоты “токсичных” мотивов (C, M, R, P‑патчи и др.).
- В FastAPI модель загружается как `mlflow.pyfunc`, вход — feature‑вектор, совпадающий с training‑pipeline.

### 2. Аллергенность
- **SOTA‑модель**: `AllergenAI` / `pLM4Alg`‑style PLM‑классификатор, предсказывающий аллергенность по аминокислотной последовательности.
- **Фичи при обучении**:
  - Deep‑learning‑эмбеддинги из protein‑language model,
  - композиционные признаки (“PILVFYW”‑патчи и др.).
- В FastAPI модель загружается из MLflow по имени `allergenicity_model` и принимает тот же feature‑вектор, что и при обучении.

### 3. Иммуногенность: B‑ и T‑клеточные эпитопы
- **B‑клеточный эпитоп**:
  - IEDB‑ and `BepiPred`‑style модель, обученная на B‑cell‑эпитопах.
  - Фичи: k‑меры (15‑mers), гибкость, содержание пролина, гидрофобные паттерны.
- **T‑клеточный эпитоп**:
  - `IEDB` / `ProPred`‑style модель для 9‑mer‑эпитопов с HLA‑аллелями.
  - Фичи: состав 9‑меров, полярные/гидрофобные “anchor‑резидуы”, аминокислотные паттерны.
- В FastAPI обе модели (`bcell_model`, `tcell_model`) загружаются из MLflow и работают в режиме “full”. В режиме “light” используются heuristic‑правила, близкие к эксперту‑ориентированным.

### 4. Стабильность антитела
- **SOTA‑подход**: `Stability Oracle` / PLM‑ + graph‑transformer‑based модель стабильности белка.
- **Фичи при обучении**:
  - ProtParam‑стиль (instability index, GRAVY, MW, pI, ароматичность, алифатический индекс),
  - структурные показатели (pLDDT, относительная доступность растворителю, delta‑pH‑стабильность) — в `README` описано как placeholder; в реальном проекте их можно заменить на реальные 3D‑фичи.
- В FastAPI модель загружается как `stability_model` из MLflow; fallback‑heuristic использует `instability_index` из ExPASy ProtParam.

### 5. Стандартные свойства белка (ProtParam‑стиль)
- Для этой группы **не используется ML‑модель**, а счетчик реализован локально через `Bio.SeqUtils.ProtParam`.
- Вычисляются:
  - молекулярная масса,
  - изоэлектрическая точка (pI),
  - индекс instability,
  - GRAVY (гидрофобность),
  - алифатический индекс,
  - half‑life,
  - средняя гибкость (flexibility),
  - полный аминокислотный состав,
  - заряд при pH 7.4.

---

## 🧩 Архитектура проекта

```text
app/
├── scoring/
│   ├── toxicity.py              # StrucToxNet‑like, MLflow
│   ├── allergenicity.py         # AllergenAI / pLM4Alg
│   ├── epitopes.py              # B‑/T‑cell эпитопы, IEDB‑style
│   ├── stability.py             # Stability Oracle‑style
│   └── protparam.py             # ProtParam‑расчёт
├── mlflow_loader.py            # реестр моделей из MLflow
├── pipeline.py                 # агрегирует все scoring‑модели
└── main.py                     # FastAPI + HTML‑фронтенд

mlflow-train/
├── train_toxicity.py
├── train_allergenicity.py
├── train_epitopes.py
├── train_stability.py          # шаблоны обучения SOTA‑моделей

static/                          # простой веб‑интерфейс
├── index.html

Dockerfile                       # FastAPI + MLflow + ML‑модели
docker-compose.yml
```

---

## ⚙️ Установка и запуск (локально)

### Требования

- Python ≥ 3.9  
- `docker` и `docker-compose` (для полного pipeline с MLflow)  
- `RDKit`, `mlflow`, `bio‑python`, `fastapi`, `uvicorn`, `transformers` (если используется PLM‑эмбеддинг)

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск без Docker

1. Поднять MLflow‑сервер локально:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns
   ```
2. Зарегистрировать модели (см. `mlflow-train/` — скрипты для `train_toxicity.py` и др.).
3. Запустить FastAPI:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. Открыть:
   - Swagger UI: `http://localhost:8000/docs`
   - Веб‑демо: `http://localhost:8000`

### 3. Запуск через Docker + docker‑compose

1. Соберите образ FastAPI:
   ```bash
   docker build -t antibody-scoring .
   ```
2. Запустите стек:
   ```bash
   docker-compose up -d
   ```
3. Перейдите в браузер:
   - FastAPI: `http://localhost:8000`
   - MLflow UI: `http://localhost:5000`

---

## 🎯 Интеграция с MLflow

Модели создаются в папке `mlflow-train/` и регистрируются в `MLflow Tracker` так:

```python
import mlflow
from mlflow.pyfunc import log_model
...
mlflow.sklearn.log_model(
    model=trained_model,
    artifact_path="model",
    registered_model_name="StrucToxNet",
    signature=signature,  # signature с feature‑вектором
)
```

FastAPI‑сервис через `app/mlflow_loader.py` загружает их из Registry:

| Модель | MLflow `registered_model_name` |
| --- | --- |
| Токсичность | `StrucToxNet` |
| Аллергенность | `AllergenAI` |
| B‑cell epitope | `BcellPred` |
| T‑cell epitope | `TcellPred` |
| Стабильность | `StabilityOracle` |

В `docker-compose.yml` и `app/main.py` задаются переменные окружения:

```ini
MLFLOW_TRACKING_URI=http://mlflow:5000
TOXICITY_MODEL_NAME=StrucToxNet
TOXICITY_MODEL_VERSION=1
ALLERGENICITY_MODEL_NAME=AllergenAI
BCELL_MODEL_NAME=BcellPred
TCELL_MODEL_NAME=TcellPred
STABILITY_MODEL_NAME=StabilityOracle
```

---

## 🌐 Размещение на бесплатном хостинге

Рекомендованные варианты:

- **Render / Railway / Fly.io**  
  - Заливай `Dockerfile` и `docker-compose.yml` как основу проекта.  
  - Включай внешний URL API (например, `https://project.onrender.com`).

- **GitHub + GitHub Pages**  
  - `static/index.html` может быть раздан через GitHub Pages как фронт‑демо,  
  - бэкенд — FastAPI‑инстанс на Render / Railway.

- **VPS (Vultr, Hetzner, DigitalOcean)**  
  - подними `docker-compose up` на сервере,
  - открой порт 8000 и настрой DNS.

---

## 🧑‍💼 Как работать с проектом

### Для ML‑инженеров

1. Обучи SOTA‑модели в `mlflow-train/`:
   - `train_toxicity.py`
   - `train_allergenicity.py`
   - `train_epitopes.py`
   - `train_stability.py`
2. Зарегистрируй модели в MLflow (`models:/StrucToxNet/1` и др.).
3. При необходимости меняй `model_name`/`version` в `docker-compose.yml` — без изменения FastAPI‑кода.

### Для DevOps / админов

- Обновляй `docker-compose.yml` при смене MLflow‑сервера или модели‑тэгов (`@champion`, `@production`).
- Логи FastAPI и MLflow можно мониторить через стандартные инструменты.

### Для биоинформатиков / заказчика

- Вводи аминокислотную последовательность в веб‑форму.
- Получаешь JSON‑ответ с:
  - рисками по токсичности, аллергенности, B/T‑эпитопам, стабильности,
  - протеин‑свойствами (ProtParam‑стиль).

---

## 🔍 Пример использования API

### Запрос

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQGLEWIGEIIGPGNGDTSYNEKFQGRATLTADKSSSTAYMELRSLTSEDSAVYFCAR...",
    "mode": "full"
  }'
```

### Ответ

```json
{
  "sequence_len": 120,
  "risk": {
    "toxicity": 0.34,
    "allergenicity": 0.21,
    "bcell_epitope": 0.45,
    "tcell_epitope": 0.38,
    "instability": 0.42,
    "overall_risk": 0.35
  },
  "properties": {
    "molecular_weight": 13500.2,
    "isoelectric_point": 7.8,
    "charge": 1.2,
    "gravy": -0.45,
    "instability_index": 38.2,
    "is_stable": true,
    "hydrophilicity": 0.45
  }
}
```

---
