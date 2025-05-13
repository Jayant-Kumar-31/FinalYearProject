from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
import numpy as np
import requests
import shap
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load model and dataset
model = joblib.load("trained_random_forest_model.joblib")
dataset = pd.read_csv('crop_dataset.csv')

# ThingSpeak API Details
THINGSPEAK_CHANNEL_ID = "2781281"
THINGSPEAK_API_KEY = "OOFGKWCQJBIKZWWG"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=1"



def fetch_realtime_data():
    response = requests.get(THINGSPEAK_URL)
    if response.status_code == 200:
        data = response.json()
        if "feeds" in data and len(data["feeds"]) > 0:
            latest_entry = data["feeds"][0]
            temperature = float(latest_entry["field1"])
            humidity = float(latest_entry["field2"])
            soil_moisture = float(latest_entry["field3"])
            ph = float(latest_entry["field4"])
            return {
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "soil_moisture": soil_moisture
            }
    return None

def get_recommendations(desired_crop, current_values):
    crop_data = dataset[dataset['Crop'].str.lower() == desired_crop.lower()]
    if crop_data.empty:
        return None
    numeric_columns = crop_data.select_dtypes(include=['float64', 'int64']).columns
    summary = crop_data[numeric_columns].agg(['min', 'max', 'mean']).T
    summary.columns = ['Ideal Min', 'Ideal Max', 'Ideal Mean']

    recommendations = []
    for param in current_values:
        if param not in summary.index:
            continue
        current = current_values[param]
        ideal_min = summary.loc[param, 'Ideal Min']
        ideal_max = summary.loc[param, 'Ideal Max']
        if current < ideal_min:
            action = 'INCREASE'
        elif current > ideal_max:
            action = 'DECREASE'
        else:
            action = 'OK'
        recommendations.append({
            'Parameter': param,
            'Current Value': round(current, 2),
            'Ideal Min': round(ideal_min, 2),
            'Ideal Max': round(ideal_max, 2),
            'Recommended Action': action
        })
    return recommendations



@app.route('/monitor')
def monitor():
    data = fetch_realtime_data()
    prediction = None
    if data:
        input_features = np.array([[data['temperature'], data['humidity'], data['ph'], data['soil_moisture']]])
        prediction = model.predict(input_features)[0]
    return render_template('monitor.html', data=data, prediction=prediction)

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/fetch-prediction')
def fetch_prediction():
    data = fetch_realtime_data()
    if data:
        input_features = np.array([[data['temperature'], data['humidity'], data['ph'], data['soil_moisture']]])
        prediction = model.predict(input_features)[0]
        return jsonify({
            "temperature": data['temperature'],
            "humidity": data['humidity'],
            "ph": data['ph'],
            "soil_moisture": data['soil_moisture'],
            "predicted_crop": prediction
        })
    else:
        return jsonify({"error": "Unable to fetch data"})

@app.route('/guidelines', methods=['GET', 'POST'])
def guidelines():
    crops = sorted(dataset['Crop'].unique())
    result = None
    current_values = None
    selected_crop = None  # <- Add this line

    if request.method == 'POST':
        selected_crop = request.form.get('crop')
        data = fetch_realtime_data()
        if data:
            current_values = {
                'pH': data['ph'],
                'Moisture': data['soil_moisture'],
                'Temperature': data['temperature'],
                'Humidity': data['humidity']
            }
            result = get_recommendations(selected_crop, current_values)

    return render_template('guidelines.html', crops=crops, result=result, crop=selected_crop)

@app.route('/shap-page')
def shap_page():
    return render_template('shap_recommend.html')

@app.route('/about')
def about_page():  # Changed function name
    return render_template('about.html')


@app.route('/shap-recommend')
def shap_recommend():
    data = fetch_realtime_data()
    if data is None:
        return jsonify({"error": "Failed to fetch sensor data"})

    # Rename for model compatibility
    data['water availability'] = data.pop('soil_moisture')

    input_df = pd.DataFrame([data])

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = [(classes[i], probabilities[i] * 100, i) for i in top_indices]

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(input_df)

    response = []

    param_key_map = {
        'Temperature': 'temperature',
        'Humidity': 'humidity',
        'pH': 'ph',
        'Moisture': 'water availability'
    }

    for crop_name, score, class_index in top_crops:
        crop_data = dataset[dataset['Crop'].str.lower() == crop_name.lower()]
        if crop_data.empty:
            continue

        ideal_ranges = crop_data[['Temperature', 'Humidity', 'pH', 'Moisture']].agg(['min', 'max']).to_dict()

        actions = []
        for param in ['Temperature', 'Humidity', 'pH', 'Moisture']:
            val = data[param_key_map[param]]
            min_val = ideal_ranges[param]['min']
            max_val = ideal_ranges[param]['max']
            if val < min_val:
                action = f"{param}: Increase to improve suitability"
            elif val > max_val:
                action = f"{param}: Decrease to improve suitability"
            else:
                action = f"{param}: {param} is already optimal"
            actions.append(action)

        response.append({
            "Crop Name": crop_name,
            "Suitability Percentage": round(score, 2),
            "Actions Needed": actions
        })

    return jsonify(response)

# Add About Route before the main block
@app.route('/about')
def about():
    print("About page route is hit")  # For debugging purposes
    return render_template('about.html')  # Ensure you have about.html in your templates folder

@app.route('/')
def home():
    return render_template('home.html')




if __name__ == "__main__":
    app.run(debug=True)