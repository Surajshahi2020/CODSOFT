import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import json  # Import json module

# Load the trained Gradient Boosting model (assuming you have saved it with pickle)
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict using the trained model
def predict(inputs):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app interface
st.title('Movie Prediction App')

# Load movie dataset with specified encoding and other CSV read parameters
try:
    movie_df = pd.read_csv('moviedataset.csv', encoding='ISO-8859-1', comment='#', quotechar='"', skipinitialspace=True)
except UnicodeDecodeError:
    st.error("There was an error loading the CSV file. Please check the file encoding.")

# Handle missing values in 'Director' column by filling NaNs with a placeholder
movie_df['Director'] = movie_df['Director'].fillna('Unknown')

# Label encoding for the Director column
director_encoder = LabelEncoder()
movie_df['Director'] = director_encoder.fit_transform(movie_df['Director'])  # Fit and transform to numerical labels

# Extract unique director names from the encoded values
directors = director_encoder.classes_.tolist()  # Get the original director names after encoding
directors.sort()  # Sort directors (optional)

# Handle genres by splitting and encoding them
movie_df['Genre'] = movie_df['Genre'].fillna('Unknown')  # Fill any missing genre values
df_genres = movie_df['Genre'].str.get_dummies(sep=', ')  # Create dummy variables for genres

# Get all possible genres, including any that might be missing from the data
all_genres = df_genres.columns.tolist()

unwanted_genres = ['Reality-TV', 'Short', 'Unknown']
all_genres = [genre for genre in all_genres if genre not in unwanted_genres]

genre_inputs = {genre: 0 for genre in all_genres}

# Streamlit input interface
year = st.number_input('Year', min_value=1900, max_value=2024, step=1)
duration = st.number_input('Duration (in minutes)', min_value=30, max_value=300, step=1)

st.subheader('Select the Genres')
columns = st.columns(5)

# Display genre checkboxes horizontally and update genre_inputs
for i, genre in enumerate(all_genres):
    col = columns[i % 5]
    if col.checkbox(f'{genre}'):
        genre_inputs[genre] = 1

votes = st.number_input('Votes', min_value=0, step=1000)

# Create a dropdown for selecting the director
director = st.selectbox('Select Director', directors)

# Transform the selected director into the encoded label
director_encoded = director_encoder.transform([director])[0]

# Prepare the input data in the format expected by the model
inputs = {
    'Year': year,
    'Duration': duration,
    'Votes': votes,
    'Director': director_encoded,  # Pass the encoded director value to the model
}

# Add genre inputs to the feature dictionary
inputs.update(genre_inputs)

# Predict the target value
if st.button('Predict'):
    prediction = predict(inputs)
    # print(inputs) 
    st.write(f'Predicted value: {prediction:.2f}')
