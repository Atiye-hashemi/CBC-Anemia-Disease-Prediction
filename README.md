

# CBC-Anemia-Disease-Prediction
Automated system that extracts CBC values from lab reports and predicts anemia types using machine learning.

# cbc-anemia-prediction

## Overview

This project presents a web application designed to predict the likelihood of Anemia based on a patient's Complete Blood Count (CBC) report (PDF upload) and demographic data (Age, Gender).

It leverages a highly-accurate XGBoost classifier trained on an Anemia dataset. Critically, it incorporates Model Explainability (SHAP) to show which CBC features (like Hemoglobin, MCV, etc.) most influenced the prediction, and provides a DeepSeek AI-powered medical summary for enhanced interpretability.

## Project Components

The project is built on two main phases:

Machine Learning Model Training: A pipeline is created and optimized using scikit-learn and XGBoost for robust binary classification (Anemia vs. No Anemia). The trained model is saved using joblib.

Flask Web Application: A web server handles PDF uploads, extracts data, makes predictions, generates SHAP explanations, and fetches an AI-driven text summary.

Set up API Key
This project uses an external AI model (DeepSeek via OpenRouter) for generating the medical explanation.

## Get an API Key from OpenRouter.ai.

Create a file named .env in the root directory of the project.

Add your API key to the file:
API_KEY="your-openrouter-api-key-here"

## Technical Details

### Model Training Pipeline (Jupyter Notebook)

The ipynb file performs the following key steps:

Data Loading & Splitting: Reads anemia_dataset.csv and splits the data into training and testing sets, using stratified sampling to maintain class balance.

Preprocessing: Implements a ColumnTransformer for:

Standard Scaling on numerical features (e.g., Hemoglobin, MCV).

Ordinal Encoding on the categorical Gender feature.

Feature Selection: Uses SelectKBest with f_classif to select the best features (currently set to use all features).

Model: Uses XGBoost Classifier.

Hyperparameter Tuning: A GridSearchCV with StratifiedKFold is used to find the optimal n_estimators, max_depth, and learning_rate for the XGBoost model.

Model Saving: The best performing pipeline is saved as xgb_model3.joblib.

### Web Application Flow (main_app.py)

The application loads the saved model (xgb_model3.joblib) and initializes the OpenAI client for the DeepSeek API.

Data Extraction: The extract_cbc_data_from_pdf function uses pdfplumber to intelligently parse tables within the uploaded PDF and extract the required CBC values (e.g., Hemoglobin, MCV, Ferritin).

Prediction: The extracted data, along with Age and Gender, is passed through the loaded model's pipeline to generate an Anemia/No Anemia prediction.

### SHAP Explanation:

The explain_with_shap function:

Creates a shap.TreeExplainer on the XGBoost model step.

Computes SHAP values.

Identifies the Top 2 most influential features based on absolute SHAP values.

Generates and saves a SHAP summary plot as shap_summary_plot.png.

### AI Explanation:

The get_ai_explanation function sends the raw input values, the prediction, and the top SHAP features to the DeepSeek API to receive a concise, clinically-relevant explanation, improving user trust and understanding.

