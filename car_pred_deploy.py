# streamlit_app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved PCA and model using pickle
with open('rlr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('pca.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

# Streamlit app
st.title('PCA-based Model Deployment')

# Input data from user (you can adjust this based on the original features of your dataset)
st.write('Input the original feature values:')
user_input = []
for i in range(67): 
    user_input.append(st.number_input(f'Feature {i+1}', min_value=-10.0, max_value=10.0, step=0.1))

# Convert the user input into a numpy array and reshape it
user_input = np.array(user_input).reshape(1, -1)

# Apply PCA to the user input
user_input_pca = pca.transform(user_input)

# Make predictions with the model
if st.button('Predict'):
    prediction = model.predict(user_input_pca)
    st.write(f'The predicted class is: {prediction[0]}')
