import streamlit as st
import joblib 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


model=joblib.load('my_model1.pkl')

# Title and Description
st.title("Rain Tomorrow Prediction")
st.write("""
### Predict whether it will rain tomorrow based on today's weather data.
Please input the following weather parameters:
""")
def preprocess_input(df):
    for col in ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']:
        df[col] = df[col].fillna(df[col].mean())

# Label Encoding for categorical columns
    le = LabelEncoder()
    
    # Encoding categorical features
    for col in [ 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']:
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string in case of any NaNs or unusual types
    

# Function to collect user input
def user_input_features():
    # Categorical Features
    location = st.selectbox('Location',(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru']))
    wind_gust_dir = st.selectbox('Wind Gust Direction',(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW']))
    wind_dir_9am = st.selectbox('Wind Direction at 9 AM',(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW']))
    wind_dir_3pm = st.selectbox('Wind Direction at 3 PM',(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW']))
    rain_today = st.selectbox('Rain Today',(['Yes', 'No']))

    # Numerical Features with default values from mean
    min_temp = st.number_input('Minimum Temperature (°C)', min_value=-50.0, max_value=50.0, value=12.194034)
    max_temp = st.number_input('Maximum Temperature (°C)', min_value=-50.0, max_value=50.0, value=23.221348)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=2.360918)
    evaporation = st.number_input('Evaporation (mm)', min_value=0.0, max_value=100.0, value=5.468232)
    sunshine = st.number_input('Sunshine (hours)', min_value=0.0, max_value=24.0, value=7.611178)
    wind_gust_speed = st.number_input('Wind Gust Speed (km/h)', min_value=0.0, max_value=150.0, value=40.035230)
    wind_speed_9am = st.number_input('Wind Speed at 9 AM (km/h)', min_value=0.0, max_value=150.0, value=14.043426)
    wind_speed_3pm = st.number_input('Wind Speed at 3 PM (km/h)', min_value=0.0, max_value=150.0, value=18.662657)
    humidity_9am = st.number_input('Humidity at 9 AM (%)', min_value=0.0, max_value=100.0, value=68.880831)
    humidity_3pm = st.number_input('Humidity at 3 PM (%)', min_value=0.0, max_value=100.0, value=51.539116)
    pressure_9am = st.number_input('Pressure at 9 AM (hPa)', min_value=800.0, max_value=1200.0, value=1017.64994)
    pressure_3pm = st.number_input('Pressure at 3 PM (hPa)', min_value=800.0, max_value=1200.0, value=1015.255889)
    cloud_9am = st.number_input('Cloud Cover at 9 AM (%)', min_value=0.0, max_value=100.0, value=4.447461)
    cloud_3pm = st.number_input('Cloud Cover at 3 PM (%)', min_value=0.0, max_value=100.0, value=4.509930)
    temp_9am = st.number_input('Temperature at 9 AM (°C)', min_value=-50.0, max_value=50.0, value=16.990631)
    temp_3pm = st.number_input('Temperature at 3 PM (°C)', min_value=-50.0, max_value=50.0, value=21.68339)
    year = st.number_input('Year', min_value=1900, max_value=2100, value=2012)
    month = st.selectbox('Month', list(range(1, 13)), index=6)  
    day = st.selectbox('Day', list(range(1, 32)), index=15)  

    # Create a DataFrame for input
    data = {
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir':wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am':wind_dir_9am,
        'WindDir3pm':wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today,
        'Year': year,
        'Month': month,
        'Day': day
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()
processed_df=preprocess_input(input_df)
processed_df=np.array(processed_df).reshape(1,-1)

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    try:
        prediction = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)

        # Assuming classes are ['No', 'Yes']
        st.subheader('Prediction')
        st.write('Rain Tomorrow:', prediction[0])

        st.subheader('Prediction Probability')
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.write(proba_df)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
