# Customer Churn Predictor

End-to-end churn prediction project with:
- Model training (`src/models/train_model.py`)
- Feature engineering (`src/features/engineer.py`)
- FastAPI inference service (`src/api/main.py`) with `/health` and `/predict`

## Requirements
- Windows PowerShell (commands below assume PowerShell)
- Python **3.11+**
- Recommended: **uv** (this repo includes `pyproject.toml` + `uv.lock`)

Install `uv` if needed:

```powershell
pip install uv
```

## Setup
From the repo root:

```powershell
cd "C:\Users\ADMIN\Downloads\DSEB65A-Group04-customer-churn-predictor"
uv sync
```

If you prefer pip:

```powershell
pip install -r requirements.txt
```

## Train model (create the sklearn Pipeline artifact)
The training script does:
- `create_features(df)` (outside the pipeline)
- fits a sklearn `Pipeline(preprocessor -> model)`
- saves the best estimator via `joblib.dump(...)`

Run:

```powershell
uv run python -m src.models.train_model --data "data\raw\train.csv" --model_dir "models" --config_out "models\config.yaml"
```

This writes (by default):
- `models\churn_model.pkl` (joblib artifact)
- `models\config.yaml`

## Make the API find the model
The API loader (`src/api/inference.py`) looks for:
- `models\model.joblib` first
- then `models\model.pkl`

### Option A (recommended): copy/rename the trained artifact

```powershell
Copy-Item "models\churn_model.pkl" "models\model.joblib"
```

### Option B: use an environment variable

```powershell
$env:MODEL_PATH = (Resolve-Path "models\churn_model.pkl")
```

## Run the FastAPI server

```powershell
uv run uvicorn src.api.main:app --reload
```

Open Swagger UI:
- `http://127.0.0.1:8000/docs`

## Test the API
### Health

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/health"
```

### Predict

```powershell
$body = @{
  age=35
  gender="Male"
  tenure=12
  usage_frequency=5
  support_calls=1
  payment_delay=0
  subscription_type="Basic"
  contract_length="Monthly"
  total_spend=1200
  last_interaction=3
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

## Notes
- Feature engineering is **stateless per row** (no lag / rolling windows). The API applies the same feature creation as training (`create_features`) before calling the saved sklearn pipeline.