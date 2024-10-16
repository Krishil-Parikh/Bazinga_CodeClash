import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


st.title('ACADEMIC SUCCESS Classifier App')

uploaded_file = st.file_uploader("Upload your CSV file (features only)", type=["csv"])

if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    
    id_column = 'id'
    
  
    if id_column in df.columns:
        features = [col for col in df.columns if col != id_column]

        if features:
            
            X = df[features]
            
            
            model = joblib.load('svm1.pkl')
            
            predictions = model.predict(X)

            submission_df = pd.DataFrame({
                'id': df[id_column],
                'Prediction': predictions
            })

            st.write("Prediction Preview:")
            st.dataframe(submission_df)

        
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
    
