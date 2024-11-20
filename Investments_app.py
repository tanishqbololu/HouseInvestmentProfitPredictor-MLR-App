

# # Set the title of Streamlit app
# st.title("Investments Profit Prediction App") 

# st.write("This app utilizes a Multi-Linear Regression model to predict investment outcomes based on various factors like marketing spend, research, promotions.")

# digital_marketing = st.number_input("Enter Spent on Digital Marketing", min_value=0)

# promotion = st.number_input("Enter Spent on Promotions", min_value=0)

# Research = st.number_input("Enter Spent on Research", min_value=0)


# cities = ["Hyderabad", "Banglore","Chennai"]

# city = st.selectbox("Select a city",cities)


# X_input = pd.DataFrame(
#     {'DigitalMarketing':[digital_marketing ],
#      'Promotion':[promotion],
#      'Research':[Research],
#      "State_Hyderabad":[0],
#      "State_Banglore":[0],
#      "State_Chennai":[0]
#     }
# )

# if city == 'Hyderabad':
#     X_input['State_Hyderabad']=1
# elif city == 'Chennai':
#     X_input['State_Chennai']=1
# elif city == 'Banglore':
#     X_input['State_Banglore']=1

# # Display the input data (for verification)
# st.write("Input data for prediction:")
# st.write(X_input)

# profit_prediction = model.predict(X_input)

# # Display the predicted profit
# if st.button('Predict'):
#     st.write(f"Predicted Profit: ${profit_prediction[0]:,.2f}")




import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Machine Learning\1. Regression\2. Linear Regression\2.Multi-Linear Regression\MLR_model_for_Investments.pkl", 'rb'))

# Get the feature names from the model
feature_names = model.feature_names_in_

# Set the title of the Streamlit app
st.title("Investments Profit Prediction App")

st.write("This app utilizes a Multi-Linear Regression model to predict investment outcomes based on various factors like marketing spend, research, promotions.")

# Collect input data from the user
digital_marketing = st.number_input("Enter Spent on Digital Marketing", min_value=0)
promotion = st.number_input("Enter Spent on Promotions", min_value=0)
research = st.number_input("Enter Spent on Research", min_value=0)

# List of cities
cities = ["Select a city","Hyderabad", "Banglore", "Chennai"]
city = st.selectbox("Select a city", cities)

# Prepare the input data for prediction
X_input = pd.DataFrame({
    'DigitalMarketing': [digital_marketing],
    'Promotion': [promotion],
    'Research': [research],
    'State_Bangalore': [0],
    'State_Chennai': [0],
    'State_Hyderabad': [0],
    "Select a city":[0]
})

# One-hot encode the selected city
if city == 'Hyderabad':
    X_input['State_Hyderabad'] = 1
elif city == 'Chennai':
    X_input['State_Chennai'] = 1
elif city == 'Banglore':
    X_input['State_Bangalore'] = 1

# Reorder the columns to match the training data columns exactly
X_input = X_input[feature_names]

# # Display the prepared input data (for verification)
# st.write("Input data for prediction:")
# st.write(X_input)

# Predict the profit using the trained model
if st.button('Predict Profit'):
    if city != "Select a city":
        profit_prediction = model.predict(X_input)
        # Display the predicted profit
        st.write(f"Predicted Profit: ${profit_prediction[0]:,.2f}")
    else:
        st.error('Give input to make predictions')
 