import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title('ACADEMIC SUCCESS Classifier App')

uploaded_file = st.file_uploader("Upload your CSV file (features only)", type=["csv"])

if uploaded_file is not None:
    # Step 2: Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    #
    id_column = 'id'
    
    # Automatically select all other columns as features
    if id_column in df.columns:
        features = [col for col in df.columns if col != id_column]

        if features:
            # Step 3: Extract features by dropping the ID column and make predictions
            X = df[features]
            
            # Load pre-trained SVM model
            model = joblib.load('svm1.pkl')
            
            predictions = model.predict(X)

            # Step 4: Create a Kaggle-style submission DataFrame
            submission_df = pd.DataFrame({
                'id': df[id_column],
                'Prediction': predictions
            })

            st.write("Prediction Preview:")
            st.dataframe(submission_df)

            # Step 5: Download the predictions as a CSV file
            csv = submission_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Submission as CSV",
                data=csv,
                file_name='submission.csv',
                mime='text/csv'
            )
        else:
            st.error("The CSV must contain at least one feature column.")
    else:
        st.error("The CSV must contain an 'id' column.")
    