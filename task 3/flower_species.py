import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('iris_flower.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app interface
st.title('Iris Flower Species Prediction App')

# Input fields for the features
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

# Features to pass to model (using DataFrame to include column names)
features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Button to make prediction
if st.button('Predict Species'):
    try:
        # Prediction using the trained model
        prediction = model.predict(features)

        # Map the predicted numeric value back to species
        species_map_reverse = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_map_reverse.get(round(prediction[0]), "Unknown")

        st.success(f"Prediction: The flower species is **{predicted_species}**!")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
