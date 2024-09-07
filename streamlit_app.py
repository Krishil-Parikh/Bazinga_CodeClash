import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib
import pickle

# Title of the app
st.title("CAR PRICE PREDICTION")

# Upload CSV file
uploaded_file = st.file_uploader("automobile_data.csv", type="csv")

if uploaded_file is not None:
    # Read CSV file into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

    # Replace '?' with NaN for proper imputation handling
    df.replace('?', np.nan, inplace=True)

    # Define features and target (replace 'target_column' with your actual target column name)
    feature_columns = [col for col in df.columns if col != 'target_column']
    

    # Separate features and target
    X = df[feature_columns]
    y = df['price']

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing pipeline for numeric data (impute and scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numeric values with mean
        ('scaler', StandardScaler())  # Scale numeric features
    ])

    # Preprocessing pipeline for categorical data (impute and encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncode categorical features
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing to features
    X_preprocessed = preprocessor.fit_transform(X)

    st.write("Preprocessed Data:")
    st.write(X_preprocessed)

    # Load the saved RandomForest model and PCA using pickle
    try:
        loaded_model = joblib.load('rlr_cleaned.pkl')  
        loaded_pca = joblib.load('pca.pkl') 
        st.write("Loaded saved model and PCA successfully!")

        # Apply PCA transformation
        X_pca = loaded_pca.transform(X_preprocessed)

        # Predict using the loaded model
        if st.button("Predict"):
            y_pred = loaded_model.predict(X_pca)
            st.write(f"Predictions: {y_pred}")

    except FileNotFoundError:
        st.error("Model or PCA file not found. Please upload the correct files.")
else:
    st.write("Please upload a CSV file to proceed.")

