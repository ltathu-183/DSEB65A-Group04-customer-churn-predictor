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

## Full run guide

### *First setup:*

**Step 0: Open project**
```bash
git clone https://github.com/chanhbui297/DSEB65A-Group04-customer-churn-predictor.git
cd DSEB65A-Group04-customer-churn-predictor
```

**Step 1: Start Kubernetes**
- Open Docker Desktop
- Make sure Kubernetes = Running

Verify:
```bash
kubectl get nodes
```
Expected:
```bash
desktop-control-plane   Ready
```

**Step 2: Deploy to Kubernetes**
```bash
kubectl apply -k k8s
```

**Step 3: Check everything**
```bash
kubectl get pods -n churn-app
```
Expected:
```bash
churn-api-xxxx  Running
churn-ui-xxxx   Running
```
**Step 4: Setup domain (only first time)**

Open hosts files:

| Operating System | File |
| :------------: | :-----------: |
| Windows | C:\Windows\System32\drivers\etc\hosts |
| macOS | /etc/hosts |
| Linux | /etc/hosts |

Add and save:

```bash
127.0.0.1 ui.churn.local
127.0.0.1 api.churn.local
```

**Step 5: Start ingress**
```bash
kubectl get pods -n ingress-nginx
```
Expected:
```bash
ingress-nginx-controller-xxxx   Running
```

**Step 6: Run the app**

Open browser:
```
http://ui.churn.local
```

### *Daily Workflow (after first setup):*
```bash
kubectl apply -k k8s
```
Then open:
```
http://ui.churn.local
```