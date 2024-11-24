import streamlit as st
import pickle
import numpy as np
import pandas as pd  # Import pandas

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The model file 'model.pkl' was not found. Please ensure it is in the correct directory.")
    st.stop()

# Set up the Streamlit app layout
st.title('Titanic Survival Prediction')

st.write("""
This app predicts whether a Titanic passenger survived based on their information.
Please fill in the details below:
""")

# Input fields for user data
Pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3], index=0)
Sex = st.selectbox("Sex", ['male', 'female'], index=0)
Age = st.number_input("Age (in years)", min_value=0, max_value=120, value=22)
SibSp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare (in dollars)", min_value=0.0, value=7.25)
Embarked = st.selectbox("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", ['C', 'Q', 'S'], index=0)

# Mapping categorical values for prediction
sex = 1 if Sex == 'female' else 0  # Convert 'female' to 1, 'male' to 0
embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]  # C -> 0, Q -> 1, S -> 2

# Define feature names as used during training
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Create the feature vector as a DataFrame
features = pd.DataFrame(
    [[Pclass, sex, Age, SibSp, Parch, Fare, embarked]],
    columns=feature_names
)

# Button to make prediction
if st.button('Predict Survival'):
    try:
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("Prediction: The passenger **survived**! 🎉")
        else:
            st.warning("Prediction: The passenger **did not survive**. 😞")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
