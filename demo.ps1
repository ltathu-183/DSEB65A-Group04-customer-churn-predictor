<#
Demo script for the DSEB65A Group 05 customer churn predictor.
Run from the repository root in PowerShell.
#>

param(
    [switch]$All,
    [switch]$Data,
    [switch]$Train,
    [switch]$Infer,
    [switch]$Deploy,
    [switch]$K8s,
    [switch]$Monitor,
    [switch]$CI
)

$WarningPreference = "SilentlyContinue"
$VerbosePreference = "SilentlyContinue"
$InformationPreference = "SilentlyContinue"

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Show-Heading($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

function Ensure-RepoRoot {
    $repoRoot = Resolve-Path .
    Write-Host "Repository root: $repoRoot"
    return $repoRoot
}

function Run-DataPipeline {
    Show-Heading 'Data Pipeline'
    Write-Host '1) Data ingestion - simulate labels' -ForegroundColor Yellow
    python .\scripts\simulate_labels.py --input data\raw\new_dataset.csv --output data\raw\new_dataset_labeled.csv

    Write-Host '2) Data merging' -ForegroundColor Yellow
    python .\scripts\merge_data.py --train data\raw\train.csv --new data\raw\new_dataset_labeled.csv --output data\raw\train.csv

    Write-Host '3) Preprocessing' -ForegroundColor Yellow
    Write-Host 'INFO - Cleaning data, handling missing values' -ForegroundColor DarkGray

    Write-Host '4) Feature engineering' -ForegroundColor Yellow
    Write-Host 'INFO - Creating feature: Age -> Age Group (Young Adult, Adult, Mid-Career, Senior)' -ForegroundColor DarkGray
    Write-Host 'INFO - Creating feature: Last Interaction -> Interaction Frequency (Highly Active, Active, Dormant)' -ForegroundColor DarkGray
    Write-Host 'INFO - Scaling numerical features (Age, Support Calls, Payment Delay, Last Interaction, Total Spend)' -ForegroundColor DarkGray
    Write-Host 'INFO - Encoding categorical features (Gender, Age Group, Interaction Frequency)' -ForegroundColor DarkGray

    python .\src\preprocess\preprocessor.py --input data\raw\train.csv --output data\preprocessed\train.csv
 
    Write-Host '[STEP 5] DVC tracking' -ForegroundColor Yellow
    dvc status

    Write-Host 'Data pipeline complete: ingestion, preprocessing, and feature engineering executed.' -ForegroundColor Green
}

function Run-Training {
    Show-Heading 'Model Training'
    
    Start-Process mlflow -ArgumentList "ui"

    Write-Host 'Training with MLflow and storing artifacts in models/' -ForegroundColor Yellow
    python .\src\models\train_model.py --data data\raw\train.csv --model_dir models --config config\drift_config.yaml --n_iter 3

    Write-Host 'Training finished. Check mlruns/ for the run and models/ for saved artifacts.' -ForegroundColor Green
}

function Start-Inference {
    Show-Heading 'Live Inference'
    Write-Host 'Starting FastAPI service in the background...' -ForegroundColor Yellow
    $fastapiProcess = Start-Process -FilePath python -ArgumentList @('-m', 'uvicorn', 'src.api.main:app', '--host', '0.0.0.0', '--port', '8000') -NoNewWindow -PassThru
    Start-Sleep -Seconds 5

    Write-Host 'Starting Streamlit app in the background...' -ForegroundColor Yellow
    $streamlitProcess = Start-Process -FilePath streamlit -ArgumentList @('run', 'streamlit_app/app.py', '--server.port=8501', '--server.address=0.0.0.0') -NoNewWindow -PassThru
    Start-Sleep -Seconds 5

    Write-Host 'Opening browsers...' -ForegroundColor Yellow
    Start-Process 'http://127.0.0.1:8000/docs'
    Start-Process 'http://127.0.0.1:8501'

    Write-Host 'Calling /predict with a sample customer payload...' -ForegroundColor Yellow
    $payload = @{
        age = 35
        gender = 'Female'
        tenure = 12
        usage_frequency = 3
        support_calls = 4
        payment_delay = 2
        subscription_type = 'Premium'
        contract_length = 'Month-to-month'
        total_spend = 410.50
        last_interaction = 5
    } | ConvertTo-Json

    Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $payload | ConvertTo-Json -Depth 4

    Write-Host 'Inference request completed. Services are running in the background.' -ForegroundColor Green
    Write-Host 'Press Enter to stop the services after reviewing output.'
    Read-Host
    Stop-Process -Id $fastapiProcess.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $streamlitProcess.Id -ErrorAction SilentlyContinue
}

function Deploy-K8s {
    Show-Heading 'Kubernetes Deployment'
    
    Write-Host 'Recreating namespace...' -ForegroundColor Yellow 
    kubectl delete namespace churn-app --ignore-not-found 
    kubectl create namespace churn-app
    
    Write-Host 'Applying Kubernetes manifests from k8s/' -ForegroundColor Yellow
    kubectl apply -k .\k8s -n churn-app

    Write-Host 'Waiting for pods to be ready...' -ForegroundColor Yellow
    kubectl wait --for=condition=ready pod --all -n churn-app --timeout=60s

    Write-Host "`n Pod status:" -ForegroundColor Cyan 
    kubectl get pods -n churn-app
    
    Write-Host "`n Starting port-forward services..." -ForegroundColor Yellow
    # Run port-forward in background 
    Start-Process powershell -ArgumentList "kubectl port-forward svc/churn-ui 8501:8501 -n churn-app" 
    Start-Process powershell -ArgumentList "kubectl port-forward svc/grafana 3000:3000 -n churn-app" 
    Start-Process powershell -ArgumentList "kubectl port-forward svc/prometheus 9090:9090 -n churn-app"
    
    Start-Sleep -Seconds 5

    Write-Host "`n Opening services in browser..." -ForegroundColor Green 
    
    Start-Process "http://localhost:8501" # Streamlit UI 
    Start-Process "http://localhost:3000" # Grafana 
    Start-Process "http://localhost:9090" # Prometheus
    
    Write-Host '`n Deployment complete!' -ForegroundColor Green
}

function Show-Monitoring {
    Show-Heading 'Monitoring and Drift Detection'
    Write-Host 'Tail the monitoring log and show drift metrics from Grafana/Prometheus.' -ForegroundColor Yellow
    Get-Content .\logs\monitoring.log -Wait
}
function Show-Monitoring {
    Show-Heading 'Monitoring and Drift Detection'

    Write-Host 'Checking monitoring pods...' -ForegroundColor Yellow
    kubectl get pods -n churn-app | Select-String "grafana|prometheus"

    Write-Host "`n Tailing monitoring log..." -ForegroundColor Yellow

    if (Test-Path .\logs\monitoring.log) {
        Start-Process powershell -ArgumentList "Get-Content .\logs\monitoring.log -Wait"
    } else {
        Write-Host " monitoring.log not found!" -ForegroundColor Red
    }

    Write-Host "`n Opening dashboards..." -ForegroundColor Yellow
    Start-Process "http://localhost:3000"   # Grafana
    Start-Process "http://localhost:9090"   # Prometheus
}


function Show-CICD {
    Show-Heading 'CI/CD Overview'

    Write-Host 'Showing workflow files:' -ForegroundColor Yellow
    
    Get-Content .github\workflows\ci.yml | Select-Object -First 20
    Write-Host '---'
    
    Get-Content .github\workflows\train.yml | Select-Object -First 20
    Write-Host '---'
    
    Get-Content .github\workflows\monitor.yml | Select-Object -First 20
    Write-Host '---'
    
    Write-Host 'Triggering workflow run...' -ForegroundColor Yellow

    if (Get-Command gh -ErrorAction SilentlyContinue) { 
        gh workflow run train.yml 
        gh workflow run monitor.yml

        Write-Host "`n Opening GitHub Actions..." -ForegroundColor Green 
        Start-Process "https://github.com/chanhbui297/DSEB65A-Group04-customer-churn-predictor/actions" 
    } else { 
        Write-Host "GitHub CLI (gh) not installed or not logged in." -ForegroundColor Red 
    }
}

# Run selected sections
Ensure-RepoRoot | Out-Null
if ($All -or $Data) { Run-DataPipeline }
if ($All -or $Train) { Run-Training }
if ($All -or $Infer) { Start-Inference }
if ($All -or $Deploy) { Start-LocalDeployment }
if ($All -or $K8s) { Deploy-K8s }
if ($All -or $Monitor) { Show-Monitoring }
if ($All -or $CI) { Show-CICD }

if (-not ($All -or $Data -or $Train -or $Infer -or $Deploy -or $K8s -or $Monitor -or $CI)) {
    Write-Host "No stage selected. Use -All or one of -Data, -Train, -Infer, -Deploy, -K8s, -Monitor, -CI" -ForegroundColor Yellow
}
