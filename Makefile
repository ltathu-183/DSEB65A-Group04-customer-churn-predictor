# --- Variables ---
# Use the same paths as defined in your repository structure
TRAIN_DATA_DVC = data/raw/train.csv.dvc
TRAIN_DATA_CSV = data/raw/train.csv
CURR_DATA_DVC  = data/raw/new_dataset.csv.dvc
CURR_DATA_CSV  = data/raw/new_dataset.csv
MODEL_DIR      = models
CONFIG_PATH    = config/drift_config.yaml

# --- Environment Setup ---
install:
	uv pip install . dvc[s3] ruff pytest pytest-cov httpx --system

# --- Data Management ---
# Pull data from DagsHub. This target ensures CSVs exist.
pull-data:
	dvc pull $(TRAIN_DATA_DVC) $(CURR_DATA_DVC)

# --- Testing & Quality Gate ---
lint:
	ruff check src/ streamlit_app/ --format=github

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# --- MLOps Workflow ---
# We make 'train' depend on 'pull-data'
train: pull-data
	echo "Running model training..."
	python src/models/train_model.py \
		--data $(TRAIN_DATA_CSV) \
		--model_dir $(MODEL_DIR) \
		--n_iter 5

# Monitoring also depends on pulling current/reference data
monitor: pull-data
	echo "Running drift detection..."
	python monitoring/detect_drift.py \
		--ref $(TRAIN_DATA_CSV) \
		--curr $(CURR_DATA_CSV) \
		--config $(CONFIG_PATH)

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage