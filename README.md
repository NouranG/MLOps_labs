# Titanic Survival Prediction — MLOps Lab

An end-to-end machine-learning project using the Kaggle Titanic dataset to practice **reproducible ML pipelines, configuration management, experiment tracking, model registration, and API deployment**.

The project compares Random Forest and LightGBM models and integrates several MLOps tools into the training and inference workflow.

## Project Objectives

- Build a configurable machine-learning pipeline.
- Apply preprocessing through reusable scikit-learn pipelines.
- Compare classification models using cross-validation.
- Track experiments and metrics with MLflow.
- Use DagsHub as the remote MLflow tracking server.
- Define a reproducible pipeline stage using DVC.
- Serve predictions through FastAPI.

## Pipeline Overview

1. **Data Loading:** Load the training and test CSV files.
2. **Feature Preparation:** Remove or transform features using configurable preprocessing.
3. **Validation:** Apply stratified cross-validation.
4. **Model Training:** Train Random Forest and LightGBM classifiers.
5. **Evaluation:** Calculate ROC-AUC using out-of-fold predictions.
6. **Experiment Tracking:** Log model parameters and metrics with MLflow.
7. **Model Registration:** Register the trained model in the MLflow model registry.
8. **Prediction:** Generate predictions for the test dataset and save them to `submission.csv`.
9. **Model Serving:** Load the registered model through FastAPI for inference.

## MLOps Tools and Technologies

| Tool | Purpose |
|---|---|
| Hydra | Configuration-driven pipeline and feature-extractor selection |
| DVC | Define the training stage and track its dependencies |
| MLflow | Experiment tracking, metric logging, and model registration |
| DagsHub | Remote MLflow tracking |
| FastAPI | Serve predictions through REST endpoints |
| scikit-learn | Preprocessing pipelines, cross-validation, and Random Forest |
| LightGBM | Gradient-boosted tree classification |
| pandas | Data loading and manipulation |
| uv | Python dependency management |

## Repository Structure

```text
MLOps_labs/
├── Src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── conf/
│   ├── config.yaml
│   ├── cfg.yaml
│   ├── cv/
│   ├── feature_extractor/
│   └── model/
├── Notebooks/
│   └── titanic.ipynb
├── dvc.yaml
├── data.dvc
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10–3.12
- uv
- Git
- Access to the Titanic dataset
- DagsHub credentials for remote experiment tracking

### Installation

```bash
git clone https://github.com/NouranG/MLOps_labs.git
cd MLOps_labs
uv sync
```

### Configure the Data

Ensure the Titanic `train.csv` and `test.csv` files are available at the paths configured in `conf/config.yaml`.

If the data is managed through DVC, configure the required remote and retrieve the tracked data using your local DVC setup.

### Configure DagsHub

Create a local `.env` file containing your DagsHub credentials:

```env
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
```

Use your own credentials and never commit the `.env` file to GitHub.

The tracking configuration is defined in `conf/cfg.yaml`.

## Run the Training Pipeline

From the repository root:

```bash
uv run python -m Src.main
```

The pipeline trains the configured candidate models, compares their ROC-AUC scores, selects a candidate, retrains it on the training data, and generates predictions for the test dataset.

The training code logs experiment metrics and registers the model through MLflow.

## Run the DVC Pipeline

```bash
uv run dvc repro
```

The configured DVC stage runs the training pipeline. Make sure the expected data files and any required remote configuration are available.

## Prediction API

The API implementation is located in `Src/predict.py`.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Predict survival for one passenger |
| POST | `/predict_batch` | Predict survival for multiple passengers |
| GET | `/health` | Check API and model-loading status |

to run the full pipeline please type python -m Src.main in the project's main directory

## API Usage

### Base URL
https://8000-01kpg6x4mnbj6w4w1fx48s4d5t.cloudspaces.litng.ai/predict

### Endpoints

#### `POST /predict`
Returns a survival prediction for a Titanic passenger.

**Request Body Example:**
```json
{
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 2,
    "Ticket": "A/5 21171",
    "Cabin": ""
}
```

**Response:**
```json
{
    "prediction": [0]
}
```
`0` = Did not survive, `1` = Survived

#### `GET /health`
Check if the API and model are running.

**Response:**
```json
{
    "status": "ok",
    "model_loaded": true
}
```

## Reproducibility and Security

- Keep dataset and model artifacts in the appropriate storage system rather than committing large files unnecessarily.
- Store credentials in environment variables, not source control.
- Keep model settings, data paths, and cross-validation settings explicit.
- Record configurations when comparing experiments.
- Treat predictions as outputs for a learning project, not as real-world risk assessments.

## Future Improvements

- Add automated tests for preprocessing, training, and API behavior.
- Add stronger input validation to prediction endpoints.
- Standardize the registered model name and version handling.
- Add CI checks and deployment automation.
- Monitor input data quality and prediction behavior.

This project demonstrates practical MLOps concepts by connecting model development, experiment tracking, artifact management, and inference into one workflow.

