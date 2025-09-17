# Streamlit UI
(https://mlopscreditcard-drt2nqzgcls3s6dkc9c5p8.streamlit.app/)
 # Credit Card Customer Segmentation (MLOps project)

This repository contains an end-to-end credit card customer segmentation project with experiments tracked using MLflow, trained models stored under `models/`, and a simple Streamlit UI to load models and make predictions.

Contents
- `app.py` - (optional) demo app entrypoint
- `src/` - training, preprocessing, and utility scripts (`train.py`, `preprocess.py`, `evaluate.py`, `utils.py`)
- `models/` - serialized models and scalers (e.g., `kmeans.pkl`, `dbscan.pkl`, `scaler.pkl`)
- `mlruns/` - MLflow experiment tracking artifacts and metrics
- `mlartifacts/` - archived artifacts from model runs
- `CC_GENERAL_preprocessed.csv` - preprocessed dataset used for modeling
- `streamlit_app.py` - minimal Streamlit UI for model selection and prediction
- `requirements.txt` - Python dependencies

Quickstart

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Credit Card Customer Segmentation (MLOps)

Comprehensive credit card customer segmentation project with MLflow experiment tracking and a Streamlit UI for quick model exploration and prediction.

This repository includes data preprocessing, clustering experiments (KMeans, Hierarchical, DBSCAN), evaluation, and a simple UI to load saved models and run single-record predictions.

Contents

- `src/` — training, preprocessing, and utility scripts (`train.py`, `preprocess.py`, `evaluate.py`, `utils.py`)
- `models/` — serialized models and scalers (e.g., `kmeans.pkl`, `dbscan.pkl`, `scaler.pkl`)
- `mlruns/` — MLflow experiment tracking artifacts and metrics (local default)
- `mlartifacts/` — archived artifacts from model runs
- `CC_GENERAL_preprocessed.csv` — preprocessed dataset used for modeling
- `streamlit_app.py` — Streamlit UI for model selection and prediction
- `CODE.ipynb`, `PREPROCESSING.ipynb`, `VISUALS.ipynb` — notebooks used for exploration and visualization
- `requirements.txt` — Python dependencies

Quickstart (Windows PowerShell)

1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Start MLflow UI to inspect runs and artifacts:

```powershell
mlflow ui --port 5000
```

Open http://127.0.0.1:5000 in your browser.

3. Start the Streamlit UI:

```powershell
streamlit run streamlit_app.py
```

What each component does

- Preprocessing: `src/preprocess.py` contains helpers to load, clean, transform and scale the data. The canonical preprocessed dataset is `CC_GENERAL_preprocessed.csv`.
- Training & experiments: `src/train.py` runs clustering experiments and logs parameters, metrics and artifacts to MLflow (`mlruns/` by default).
- Models: trained models and a scaler are saved under `models/` as joblib pickles (e.g. `kmeans.pkl`, `scaler.pkl`).
- UI: `streamlit_app.py` loads the pickled model and scaler and allows single-record prediction with the same feature order used for training.

Model & input format

- The preprocessing pipeline and preprocessed CSV contain the following numeric columns (order used by the Streamlit UI):

```
CUST_ID (dropped for modeling),
BALANCE,
BALANCE_FREQUENCY,
PURCHASES,
ONEOFF_PURCHASES,
INSTALLMENTS_PURCHASES,
CASH_ADVANCE,
PURCHASES_FREQUENCY,
ONEOFF_PURCHASES_FREQUENCY,
PURCHASES_INSTALLMENTS_FREQUENCY,
CASH_ADVANCE_FREQUENCY,
CASH_ADVANCE_TRX,
PURCHASES_TRX,
CREDIT_LIMIT,
PAYMENTS,
MINIMUM_PAYMENTS,
PRC_FULL_PAYMENT,
TENURE
```

- The Streamlit UI and example code expect a `scaler.pkl` (joblib) if scaling was applied at training time. If present, the app will apply the scaler before prediction.

Using MLflow

- To view runs and artifacts locally, start the MLflow UI (see Quickstart).
- `src/train.py` logs runs to the default tracking URI (local `mlruns/`). You can change tracking server or backend store via environment variables (see MLflow docs):

```powershell
set MLFLOW_TRACKING_URI=http://your-tracking-server:5000
```

- To promote models to a registry or to load MLflow model format instead of pickles, adapt `src/train.py` to call `mlflow.sklearn.log_model(..., registered_model_name='CreditCardClustering')`.

Streamlit UI details

- The UI lists `.pkl` files from `models/`. Select a model and enter numeric values in the sidebar form.
- If `scaler.pkl` exists it will be applied automatically.
- The app attempts to query MLflow runs related to the model name (best-effort) and will not fail if no tracking server is reachable.

Recommended workflow

1. Run experiments with `src/train.py`. Confirm metrics and artifacts in MLflow UI.
2. Save the best model and the scaler to `models/` (joblib):

```python
import joblib
joblib.dump(best_model, 'models/kmeans.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
```

3. Use `streamlit_app.py` for quick manual testing or demos.

Extending this project

- Add batch inference via CSV upload in Streamlit.
- Add MLflow Model Registry integration and automated promotion pipelines.
- Containerize the Streamlit app and MLflow tracking server with Docker for reproducible deployment.

Dependencies

The project already includes `mlflow` and `streamlit` in `requirements.txt`. Key packages:

- pandas, numpy, scikit-learn, joblib, mlflow, streamlit

Troubleshooting

- If Streamlit cannot find models, ensure your `models/` directory contains `.pkl` files and is in the repo root.
- If MLflow UI shows no runs, verify `src/train.py` ran and logged runs (check `mlruns/`).

License

This repository is provided as-is. Modify and extend as needed.
