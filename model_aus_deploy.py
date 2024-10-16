import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


location_mapping = {
    'Albury': 0,
    'BadgerysCreek': 1,
    'Cobar': 2,
    'CoffsHarbour': 3,
    'Moree': 4,
    'Newcastle': 5,
    'NorahHead': 6,
    'NorfolkIsland': 7,
    'Penrith': 8,
    'Richmond': 9,
    'Sydney': 10,
    'SydneyAirport': 11,
    'WaggaWagga': 12,
    'Williamtown': 13,
    'Wollongong': 14,
    'Canberra': 15,
    'Tuggeranong': 16,
    'MountGinini': 17,
    'Ballarat': 18,
    'Bendigo': 19,
    'Sale': 20,
    'MelbourneAirport': 21,
    'Melbourne': 22,
    'Mildura': 23,
    'Nhil': 24,
    'Portland': 25,
    'Watsonia': 26,
    'Dartmoor': 27,
    'Brisbane': 28,
    'Cairns': 29,
    'GoldCoast': 30,
    'Townsville': 31,
    'Adelaide': 32,
    'MountGambier': 33,
    'Nuriootpa': 34,
    'Woomera': 35,
    'Albany': 36,
    'Witchcliffe': 37,
    'PearceRAAF': 38,
    'PerthAirport': 39,
    'Perth': 40,
    'SalmonGums': 41,
    'Walpole': 42,
    'Hobart': 43,
    'Launceston': 44,
    'AliceSprings': 45,
    'Darwin': 46,
    'Katherine': 47,
    'Uluru': 48
}

wind_gust_dir_mapping = {
    'W': 0, 'WNW': 1, 'WSW': 2, 'NE': 3, 'NNW': 4, 'N': 5,
    'NNE': 6, 'SW': 7, 'ENE': 8, 'SSE': 9, 'S': 10,
    'NW': 11, 'SE': 12, 'ESE': 13, 'E': 14, 'SSW': 15
}


wind_dir_9am_mapping = wind_gust_dir_mapping
wind_dir_3pm_mapping = wind_gust_dir_mapping

rain_today_mapping = {
    'No': 0,
    'Yes': 1
}


model = joblib.load('my_model21.pkl')
scaler = joblib.load('scalerr.pkl')  


try:
    label_encoders = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    label_encoders = None
    st.warning("Label encoders not found. Please ensure 'label_encoders.pkl' is available.")


st.title("Rain Tomorrow Prediction")
st.write("""
### Predict whether it will rain tomorrow based on today's weather data.
Please input the following weather parameters:
""")

def preprocess_input(df):
   
    numerical_cols = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day'
    ]
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    
    df['Location'] = df['Location'].map(location_mapping)
    df['WindGustDir'] = df['WindGustDir'].map(wind_gust_dir_mapping)
    df['WindDir9am'] = df['WindDir9am'].map(wind_dir_9am_mapping)
    df['WindDir3pm'] = df['WindDir3pm'].map(wind_dir_3pm_mapping)
    df['RainToday'] = df['RainToday'].map(rain_today_mapping)
    
    
    df.fillna(-1, inplace=True)
    
    
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df


def user_input_features():
 
    location = st.selectbox('Location',(
        'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
        'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
        'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
        'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
        'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
        'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
        'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
        'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
        'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
        'AliceSprings', 'Darwin', 'Katherine', 'Uluru'))
    
    wind_gust_dir = st.selectbox('Wind Gust Direction',(
        'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
        'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
    
    wind_dir_9am = st.selectbox('Wind Direction at 9 AM',(
        'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
        'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
    
    wind_dir_3pm = st.selectbox('Wind Direction at 3 PM',(
        'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
        'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
    
    rain_today = st.selectbox('Rain Today',(['Yes', 'No']))

    
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

   
    data = {
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
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


input_df = user_input_features()


st.subheader('User Input Features')
st.write(input_df)

processed_df = preprocess_input(input_df)


st.subheader('Processed Input Features')
st.write(processed_df)


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
