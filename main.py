import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pickle
import pandas as pd
import os
from datetime import date
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler from pkl files
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'rainfall.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models', 'scale.pkl'), 'rb'))

# Define feature names (numeric features used in training)
NUMERIC_FEATURES = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'year', 'month', 'day']

# Categorical feature columns after one-hot encoding (from training - 110 total features)
LOCATION_FEATURES = [
    'Location_Albany', 'Location_Albury', 'Location_AliceSprings', 'Location_BadgerysCreek',
    'Location_Ballarat', 'Location_Bendigo', 'Location_Brisbane', 'Location_Cairns',
    'Location_Canberra', 'Location_Cobar', 'Location_CoffsHarbour', 'Location_Dartmoor',
    'Location_Darwin', 'Location_Delhi', 'Location_GoldCoast', 'Location_Hobart',
    'Location_Katherine', 'Location_Launceston', 'Location_Melbourne', 'Location_MelbourneAirport',
    'Location_Mildura', 'Location_Moree', 'Location_MountGambier', 'Location_MountGinini',
    'Location_Newcastle', 'Location_Nhil', 'Location_NorahHead', 'Location_NorfolkIsland',
    'Location_Nuriootpa', 'Location_PearceRAAF', 'Location_Penrith', 'Location_Perth',
    'Location_PerthAirport', 'Location_Portland', 'Location_Richmond', 'Location_Sale',
    'Location_SalmonGums', 'Location_Sydney', 'Location_SydneyAirport', 'Location_Townsville',
    'Location_Tuggeranong', 'Location_Uluru', 'Location_WaggaWagga', 'Location_Walpole',
    'Location_Watsonia', 'Location_Williamtown', 'Location_Witchcliffe', 'Location_Wollongong',
    'Location_Woomera'
]

RAIN_TODAY_FEATURES = ['RainToday_Yes']

WIND_GUST_DIR_FEATURES = [
    'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE',
    'WindGustDir_NNE', 'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_S',
    'WindGustDir_SE', 'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW',
    'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW'
]

WIND_DIR_9AM_FEATURES = [
    'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NE',
    'WindDir9am_NNE', 'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_S',
    'WindDir9am_SE', 'WindDir9am_SSE', 'WindDir9am_SSW', 'WindDir9am_SW',
    'WindDir9am_W', 'WindDir9am_WNW', 'WindDir9am_WSW'
]

WIND_DIR_3PM_FEATURES = [
    'WindDir3pm_ENE', 'WindDir3pm_ESE', 'WindDir3pm_N', 'WindDir3pm_NE',
    'WindDir3pm_NNE', 'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S',
    'WindDir3pm_SE', 'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW',
    'WindDir3pm_W', 'WindDir3pm_WNW', 'WindDir3pm_WSW'
]

CATEGORICAL_ENCODED_FEATURES = (LOCATION_FEATURES + RAIN_TODAY_FEATURES + 
                                 WIND_GUST_DIR_FEATURES + WIND_DIR_9AM_FEATURES + 
                                 WIND_DIR_3PM_FEATURES)

# All feature columns in order (110 total features)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_ENCODED_FEATURES

# Location options for the dropdown
LOCATIONS = [
    'Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat',
    'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour',
    'Dartmoor', 'Darwin', 'Delhi', 'GoldCoast', 'Hobart', 'Katherine',
    'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree',
    'MountGambier', 'MountGinini', 'Newcastle', 'Nhil', 'NorahHead',
    'NorfolkIsland', 'Nuriootpa', 'PearceRAAF', 'Penrith', 'Perth',
    'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums', 'Sydney',
    'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga',
    'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera'
]


def load_default_values():
    numeric_defaults = {feat: 0.0 for feat in NUMERIC_FEATURES}
    categorical_defaults = {
        'WindGustDir': 'N',
        'WindDir9am': 'N',
        'WindDir3pm': 'N',
        'RainToday': 'No',
        'Location': LOCATIONS[0] if LOCATIONS else None,
    }

    dataset_path = os.path.join(BASE_DIR, 'Weather-Dataset.csv')
    if not os.path.exists(dataset_path):
        return numeric_defaults, categorical_defaults

    try:
        df = pd.read_csv(dataset_path)

        if 'Date' in df.columns:
            date_series = pd.to_datetime(df['Date'], errors='coerce')
            df = df.copy()
            df['year'] = date_series.dt.year
            df['month'] = date_series.dt.month
            df['day'] = date_series.dt.day

        numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]
        if numeric_cols:
            means = df[numeric_cols].mean(numeric_only=True)
            for col in numeric_cols:
                value = means.get(col)
                if pd.notna(value):
                    numeric_defaults[col] = float(value)

        def mode_or(series, fallback):
            if series is None:
                return fallback
            series = series.dropna()
            if series.empty:
                return fallback
            return series.mode().iloc[0]

        if 'WindGustDir' in df.columns:
            categorical_defaults['WindGustDir'] = mode_or(df['WindGustDir'], categorical_defaults['WindGustDir'])
        if 'WindDir9am' in df.columns:
            categorical_defaults['WindDir9am'] = mode_or(df['WindDir9am'], categorical_defaults['WindDir9am'])
        if 'WindDir3pm' in df.columns:
            categorical_defaults['WindDir3pm'] = mode_or(df['WindDir3pm'], categorical_defaults['WindDir3pm'])
        if 'RainToday' in df.columns:
            categorical_defaults['RainToday'] = mode_or(df['RainToday'], categorical_defaults['RainToday'])
        if 'Location' in df.columns:
            categorical_defaults['Location'] = mode_or(df['Location'], categorical_defaults['Location'])
    except Exception:
        return numeric_defaults, categorical_defaults

    return numeric_defaults, categorical_defaults


DEFAULT_NUMERIC_VALUES, DEFAULT_CATEGORICAL_VALUES = load_default_values()


def get_float_value(form, key, default):
    raw_value = form.get(key)
    if raw_value is None or raw_value == '':
        return float(default)
    try:
        return float(raw_value)
    except (ValueError, TypeError):
        return float(default)


def get_int_value(form, key, default):
    raw_value = form.get(key)
    if raw_value is None or raw_value == '':
        return int(default)
    try:
        return int(float(raw_value))
    except (ValueError, TypeError):
        return int(default)


@app.route('/')
def home():
    """Route to display the home page with the prediction form"""
    return render_template('index.html', locations=LOCATIONS)


@app.route('/predict', methods=['POST'])
def predict():
    """Route to handle prediction requests"""
    try:
        # Get form values
        location = request.form.get('Location') or DEFAULT_CATEGORICAL_VALUES['Location']

        min_temp = get_float_value(request.form, 'MinTemp', DEFAULT_NUMERIC_VALUES['MinTemp'])
        max_temp = get_float_value(request.form, 'MaxTemp', DEFAULT_NUMERIC_VALUES['MaxTemp'])
        rainfall = get_float_value(request.form, 'Rainfall', DEFAULT_NUMERIC_VALUES['Rainfall'])
        wind_gust_speed = get_float_value(request.form, 'WindGustSpeed', DEFAULT_NUMERIC_VALUES['WindGustSpeed'])
        wind_speed_9am = get_float_value(request.form, 'WindSpeed9am', DEFAULT_NUMERIC_VALUES['WindSpeed9am'])
        wind_speed_3pm = get_float_value(request.form, 'WindSpeed3pm', DEFAULT_NUMERIC_VALUES['WindSpeed3pm'])
        humidity_9am = get_float_value(request.form, 'Humidity9am', DEFAULT_NUMERIC_VALUES['Humidity9am'])
        humidity_3pm = get_float_value(request.form, 'Humidity3pm', DEFAULT_NUMERIC_VALUES['Humidity3pm'])
        pressure_9am = get_float_value(request.form, 'Pressure9am', DEFAULT_NUMERIC_VALUES['Pressure9am'])
        pressure_3pm = get_float_value(request.form, 'Pressure3pm', DEFAULT_NUMERIC_VALUES['Pressure3pm'])

        temp_9am_default = DEFAULT_NUMERIC_VALUES['Temp9am']
        temp_3pm_default = DEFAULT_NUMERIC_VALUES['Temp3pm']
        if max_temp >= min_temp:
            temp_range = max_temp - min_temp
            temp_9am_default = min_temp + temp_range * 0.35
            temp_3pm_default = min_temp + temp_range * 0.8

        temp_9am = get_float_value(request.form, 'Temp9am', temp_9am_default)
        temp_3pm = get_float_value(request.form, 'Temp3pm', temp_3pm_default)

        today = date.today()
        year = get_int_value(request.form, 'year', today.year)
        month = get_int_value(request.form, 'month', today.month)
        day = get_int_value(request.form, 'day', today.day)

        rain_today = request.form.get('RainToday')
        if rain_today not in ('Yes', 'No'):
            rain_today = 'Yes' if rainfall > 0 else 'No'

        wind_gust_dir = request.form.get('WindGustDir') or DEFAULT_CATEGORICAL_VALUES['WindGustDir']
        wind_dir_9am = request.form.get('WindDir9am') or DEFAULT_CATEGORICAL_VALUES['WindDir9am']
        wind_dir_3pm = request.form.get('WindDir3pm') or DEFAULT_CATEGORICAL_VALUES['WindDir3pm']

        # Create numeric features array
        numeric_values = [min_temp, max_temp, rainfall, wind_gust_speed,
                          wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm,
                          pressure_9am, pressure_3pm, temp_9am, temp_3pm, year, month, day]
        
        # Scale numeric features
        numeric_scaled = scaler.transform([numeric_values])[0]
        
        # Create a dictionary for all features initialized with scaled numeric values
        feature_dict = {feat: val for feat, val in zip(NUMERIC_FEATURES, numeric_scaled)}
        
        # Initialize all categorical encoded features to 0
        for cat_feat in CATEGORICAL_ENCODED_FEATURES:
            feature_dict[cat_feat] = 0
        
        # Set the appropriate one-hot encoded values
        # Location encoding
        loc_col = f'Location_{location}'
        if loc_col in feature_dict:
            feature_dict[loc_col] = 1
        
        # RainToday encoding
        if rain_today == 'Yes':
            feature_dict['RainToday_Yes'] = 1
        
        # Wind direction encodings
        gust_col = f'WindGustDir_{wind_gust_dir}'
        if gust_col in feature_dict:
            feature_dict[gust_col] = 1
            
        dir9am_col = f'WindDir9am_{wind_dir_9am}'
        if dir9am_col in feature_dict:
            feature_dict[dir9am_col] = 1
            
        dir3pm_col = f'WindDir3pm_{wind_dir_3pm}'
        if dir3pm_col in feature_dict:
            feature_dict[dir3pm_col] = 1
        
        # Create DataFrame with all features in correct order
        input_data = pd.DataFrame([[feature_dict[feat] for feat in ALL_FEATURES]], columns=ALL_FEATURES)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get probability percentages
        no_rain_prob = round(prediction_proba[0] * 100, 1)
        rain_prob = round(prediction_proba[1] * 100, 1)
        
        # Render appropriate template based on prediction
        if prediction == 1:
            return render_template('change.html', probability=rain_prob)
        else:
            return render_template('nochange.html', probability=no_rain_prob)
            
    except Exception as e:
        return render_template('index.html', error=str(e), locations=LOCATIONS)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
