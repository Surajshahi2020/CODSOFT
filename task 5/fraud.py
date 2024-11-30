import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('fraud_detection.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app interface
st.title('Fraud Detection App')

# Time input
time = st.number_input("Time", min_value=0, max_value=15012, value=100)

# Input fields for V1 to V28
input_data = []
for i in range(1, 29):  # For V1 to V28
    input_value = st.number_input(f"V{i}", min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
    input_data.append(input_value)

# Add the Time feature to the input data
input_data.insert(0, time)  # Insert 'time' at the start of the list

# Create a DataFrame for prediction
features = pd.DataFrame([input_data], columns=['Time'] + [f'V{i}' for i in range(1, 29)])

# Prediction button
if st.button('Predict Fraud'):
    try:
        # Prediction using the trained model
        prediction = model.predict(features)

        # Map the predicted numeric value to fraud detection (0 = Genuine, 1 = Fraudulent)
        fraud_label = "Fraudulent" if prediction[0] == 1 else "Genuine"

        st.success(f"Prediction: The transaction is **{fraud_label}**!")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
