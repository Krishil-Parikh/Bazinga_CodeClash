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


st.title("CAR PRICE PREDICTION")


uploaded_file = st.file_uploader("automobile_data.csv", type="csv")

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

    
    df.replace('?', np.nan, inplace=True)

   
    feature_columns = [col for col in df.columns if col != 'target_column']
    

    
    X = df[feature_columns]
    y = df['price']

    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler()) 
    ])

   
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  
    ])

   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

  
    X_preprocessed = preprocessor.fit_transform(X)

    st.write("Preprocessed Data:")
    st.write(X_preprocessed)

    
    try:
        loaded_model = joblib.load('rlr_cleaned.pkl')  
        loaded_pca = joblib.load('pca.pkl') 
        st.write("Loaded saved model and PCA successfully!")

        X_pca = loaded_pca.transform(X_preprocessed)

        
        if st.button("Predict"):
            y_pred = loaded_model.predict(X_pca)
            st.write(f"Predictions: {y_pred}")

    except FileNotFoundError:
        st.error("Model or PCA file not found. Please upload the correct files.")
else:
    st.write("Please upload a CSV file to proceed.")

