# Rent Price Prediction - Machine Learning Project

This project focuses on predicting house rent prices using a machine learning model trained on real-world data. The dataset originates from various cities in India and contains information about the property type, location, size, number of rooms, furnishing status, and other relevant features.

## Objective

To build a machine learning model that can accurately estimate the rent price of a house in India based on key features provided by the user. The goal is to practice and understand the complete ML workflow, from data cleaning and preprocessing to model training, evaluation, and deployment.

## Dataset

The dataset used is publicly available and contains rental listings from cities across India. Each row in the dataset represents a rental listing and includes the following key features:

- BHK (Number of bedrooms)
- Size (Square feet)
- City
- Furnishing Status (Furnished, Semi-Furnished, Unfurnished)
- Bathroom (Number of bathrooms)
- Rent (Target variable)

Irrelevant columns such as `Posted On` and `Tenant Preferred` were removed to simplify the model and focus on features most relevant to pricing.

## Project Workflow

### 1. Data Exploration and Cleaning
- Loaded and inspected the dataset
- Verified there were no missing values
- Removed non-essential columns
- Renamed some columns for clarity (e.g., `BHK` as Bedrooms)

### 2. Feature Engineering
- Categorical features (`City`, `Furnishing Status`) were encoded using OneHotEncoder via `ColumnTransformer`
- Numerical features were passed through without modification

### 3. Model Training
- Applied Linear Regression using scikit-learn
- Split data into training and test sets
- Evaluated model performance using R² Score and Mean Squared Error

### 4. Model Saving
- Trained model (and pipeline) was saved using joblib for future reuse

### 5. Web Application
- A basic web interface was developed using Streamlit
- Users can input property details to receive a predicted rent estimate
- The Streamlit app reads user input, processes it through the trained model, and displays the predicted rent




