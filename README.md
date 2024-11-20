# House Investment Profit Predictor - MLR App
This repository contains two key components for predicting profit from house investments using a Multiple Linear Regression (MLR) model:

* house_investment_predictor.py:
    This Python script implements the core machine learning model using Multiple Linear Regression. It loads a dataset, preprocesses the data (including One-Hot Encoding for categorical variables), trains the model, performs feature selection using Ordinary Least Squares (OLS) regression, and evaluates the model's performance. The trained model is then saved as a pickle file for later use.

* streamlit_app.py:
    This script provides an interactive web interface using Streamlit. It allows users to input values such as marketing spend, promotions, and research spend for different cities. The app then loads the trained model, processes the inputs, and predicts the potential profit from real estate investments. The result is displayed dynamically as the user interacts with the app.

* Key Features:
  * Data Preprocessing: Handles categorical variable encoding and splitting the dataset into training and testing sets.
  * Multiple Linear Regression Model: Predicts investment profit based on multiple factors.
  * Feature Selection: Identifies and removes insignificant features using p-value analysis.
  * Streamlit Interface: A user-friendly web app that allows interactive input and real-time profit predictions.

* How to Use:
  * Training the Model: Run the house_investment_predictor.py to train and save the model.
  * Running the App: Run the streamlit_app.py to launch the web interface and make prediction
