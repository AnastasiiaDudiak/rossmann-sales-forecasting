# Rossmann Sales Forecasting

## Project Overview

Rossmann Sales Forecasting is a machine learning application for analyzing, modeling, and forecasting daily retail store sales.

The system combines a Streamlit frontend and a FastAPI backend to provide an interactive environment for data exploration, model training, evaluation, comparison, management, and sales forecasting.

The application also includes a Jupyter Notebook REST client for interacting with the API and performing scenario-based analysis.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- TensorFlow / Keras
- FastAPI
- Streamlit
- Plotly
- Requests
- Jupyter Notebook
- Pytest

## Dataset

The project uses the Rossmann Store Sales dataset.

The prepared dataset contains historical daily sales data for 100 stores.

The data is divided chronologically to prevent data leakage and ensure realistic evaluation of forecasting performance.

**Data splitting:**

- Training: 2013–2014
- Validation: January–March 2015
- Test: April–July 2015

The dataset includes information about:

- Daily sales
- Store identifiers
- Promotional activity
- Public holidays
- School holidays
- Store opening status
- Calendar features
- Store characteristics

The `Customers` column is used for exploratory data analysis only. It is excluded from model training to prevent data leakage because actual customer counts are not known in advance when forecasting future sales.

## Machine Learning Models

Three machine learning algorithms are implemented:

1. Random Forest Regressor
2. LightGBM Regressor
3. Keras Multilayer Perceptron (MLP)

The models use promotional activity, holidays, calendar features, and store characteristics to predict daily sales.

### Target Transformation

The target variable is transformed using:

`log1p(Sales)`

Predictions are converted back to the original sales scale using:

`expm1(prediction)`

For closed stores, the predicted sales value is set to zero.

### Model Configuration

The application supports:

- Algorithm selection
- Feature selection
- Hyperparameter configuration
- Model training
- Model persistence
- Model metadata inspection
- Comparison of trained models

## Application Features

### 1. Data Exploration

The Data Exploration page provides interactive analysis of historical retail sales.

Features include:

- Sales distributions
- Daily, weekly, and monthly sales trends
- Store comparisons
- Promotional activity analysis
- Holiday analysis
- Interactive visualizations

### 2. Model Training

The Model Training page allows users to configure and train machine learning models.

Features include:

- Algorithm selection
- Feature selection
- Hyperparameter configuration
- Training through FastAPI
- Training and validation metrics
- Model persistence
- Saved training curves for LightGBM and Keras MLP

### 3. Model Comparison

The Model Comparison page allows users to evaluate and compare saved models.

Features include:

- Comparison of multiple trained models
- Train, Validation, and Test evaluation periods
- MAE, RMSE, WAPE, R², and Log RMSE
- Baseline comparison
- Interactive comparison charts
- Model configuration inspection
- Feature importance for tree-based models

### 4. Sales Forecasting

The Sales Forecasting page provides an interactive forecasting interface.

Features include:

- Store selection
- Date range selection
- Model selection
- Actual versus predicted sales
- Forecast evaluation
- Interactive forecasting charts
- CSV export
- What-if scenario analysis
- Promotional scenario comparison

## System Architecture

The application follows a client-server architecture.

The Streamlit frontend and Jupyter REST client communicate with the FastAPI backend using HTTP requests.

**Architecture:**

```text
           Streamlit Frontend
                   |
                   | HTTP Requests
                   |
                   v
             FastAPI Backend
                   ^
                   |
                   | HTTP Requests
                   |
           Jupyter REST Client

             FastAPI Backend
                   |
                   v
        Machine Learning Services
                   |
                   v
        Saved Models and Metadata
```

**Component responsibilities:**

- Streamlit provides the interactive user interface, data exploration, model comparison, and forecasting.
- FastAPI handles model training, inference, forecasting, and model management.
- Machine learning services handle preprocessing, feature engineering, model training, and prediction.
- Jupyter Notebook provides a separate REST client for API interaction and scenario analysis.
- Saved models and metadata support model persistence and reuse.

## Project Structure

```text
app_sales_forecasting/
|
|-- api/
|   |-- main.py
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- feature_importance.py
|   |-- model_training.py
|   |-- mlp_training.py
|   |-- model_service.py
|   |-- preprocessing.py
|   |-- training_service.py
|
|-- streamlit_app/
|   |-- app.py
|   |-- pages/
|       |-- 1_Data_Exploration.py
|       |-- 2_Model_Training.py
|       |-- 3_Model_Comparison.py
|       |-- 4_Forecasting.py
|
|-- notebooks/
|   |-- api_client.ipynb
|
|-- data/
|
|-- models/
|
|-- artifacts/
|
|-- tests/
|   |-- test_api.py
|
|-- requirements.txt
|-- README.md
|-- .env.example
```

## Installation

### 1. Open the Project Directory

Open a terminal in the project root directory:

```bash
cd app_sales_forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

The project includes a `.env.example` file containing an example environment configuration.

Review this file and configure the environment as required by the application.

## Running the Application

Run all commands from the project root directory.

### Start FastAPI

Open a terminal and execute:

```bash
uvicorn api.main:app --reload
```

The FastAPI backend will be available at:

http://127.0.0.1:8000

Interactive Swagger documentation:

http://127.0.0.1:8000/docs

### Start Streamlit

Open a separate terminal and execute:

```bash
streamlit run streamlit_app/app.py
```

The Streamlit application will be available at:

http://localhost:8501

Both FastAPI and Streamlit should remain running while using the application.

## REST API

FastAPI provides endpoints for accessing dataset information, managing trained models, generating predictions, and performing forecasting.

### API Operations

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| GET | `/data/info` | Retrieve dataset information |
| GET | `/models` | List saved models |
| GET | `/models/{name}` | Retrieve model information and metrics |
| DELETE | `/models/{name}` | Delete a saved model |
| POST | `/predict` | Generate a single prediction |
| POST | `/predict/batch` | Generate multiple predictions |
| POST | `/forecast` | Generate sales forecasts |
| POST | `/forecast/scenario` | Generate forecasts under selected conditions |

The API also supports model training through the application.

All available API operations and their request schemas can be inspected using Swagger UI.

### API Documentation

Open:

http://127.0.0.1:8000/docs

Swagger UI allows users to inspect endpoints, submit requests, and view responses.

## Jupyter REST Client

The project includes a Jupyter Notebook REST client:

`notebooks/api_client.ipynb`

The notebook demonstrates interaction with the FastAPI backend using HTTP requests.

### Supported Operations

- Check API health
- Retrieve dataset information
- Retrieve available models
- Generate a single prediction
- Generate batch predictions
- Generate forecasts for selected stores and dates
- Generate scenario forecasts
- Compare promotional scenarios
- Visualize forecasting results

### Running the Notebook

First, start the FastAPI backend.

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

`notebooks/api_client.ipynb`

Execute the notebook cells in order.

The notebook sends requests to the local FastAPI server at:

http://127.0.0.1:8000

### Scenario Comparison

The REST client supports comparison of forecasts with and without promotional activity.

The results include:

- Predicted sales without promotion
- Predicted sales with promotion
- Absolute difference
- Percentage difference
- Interactive comparison chart

## Model Evaluation

The following evaluation metrics are used:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **WAPE:** Weighted Absolute Percentage Error
- **R²:** Coefficient of Determination
- **Log RMSE:** Root Mean Squared Error in logarithmic target space

Model performance is evaluated separately on the training, validation, and test datasets.

### Baseline Comparison

Machine learning models are compared with a baseline forecasting approach.

This allows users to evaluate whether the trained models improve forecasting accuracy over a simpler reference method.

### Model Performance

The following results were obtained on the test dataset:

| Model | Test WAPE |
|---|---:|
| Baseline | 20.87% |
| Random Forest | 11.86% |
| LightGBM | 11.79% |
| Keras MLP | 11.92% |

Among these evaluated configurations, LightGBM achieved the lowest Test WAPE.

## Training Curves

The application includes saved training visualizations for LightGBM and Keras MLP models.

Training curves are available through the Model Training page.

They allow users to inspect the training process and analyze model behavior.

## Feature Importance

Feature importance is available for Random Forest and LightGBM models.

The analysis helps identify which features contribute most strongly to model predictions.

Feature importance does not establish causality.

## What-if Scenario Analysis

The application supports scenario-based forecasting.

Users can investigate how predicted sales change under different conditions, including:

- Promotional activity
- School holidays
- Public holidays
- Store opening status

The system allows users to compare predictions across scenarios and visualize the differences.

Scenario results represent model-based estimates rather than guaranteed real-world outcomes or causal effects.

## Testing

The project includes automated tests implemented with Pytest.

Run the tests from the project root directory:

```bash
pytest tests/ -v
```

The latest recorded test run completed with **18 passing tests**.

The tests cover API behavior and key forecasting operations, including:

- API health checks
- Dataset information
- Saved model listing
- Model metadata and evaluation metrics
- Unknown model handling
- Forecast generation
- Date range validation
- Scenario forecasting
- Closed-store predictions

Additional manual API verification can be performed using Swagger UI.

## Project Results

The project demonstrates a complete machine learning workflow for retail sales forecasting.

The implemented system combines:

- Historical sales analysis
- Data preprocessing and feature engineering
- Multiple machine learning algorithms
- Chronological model evaluation
- Model comparison
- REST API integration
- Interactive Streamlit application
- Jupyter REST client
- Scenario-based forecasting

The system provides an interactive environment for studying sales patterns, evaluating forecasting models, and exploring possible changes in future retail sales.

## Conclusion

Rossmann Sales Forecasting demonstrates the integration of machine learning, data analysis, API development, and interactive visualization into a single application.

The project can be used as an educational and portfolio example of an end-to-end machine learning system for retail sales forecasting.