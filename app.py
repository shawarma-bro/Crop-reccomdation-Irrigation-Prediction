from flask import Flask, render_template, request, jsonify, flash
import pickle
import numpy as np
import requests

app = Flask(__name__)
app.secret_key = "supersecret"

# Load ML models
with open("models/crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("models/label_encoder_crop.pkl", "rb") as f:
    crop_le = pickle.load(f)

with open("models/irrigation_model.pkl", "rb") as f:
    irrigation_model = pickle.load(f)

# Terrain coordinates for weather API
terrain_coords = {
    "Coastal": {"lat": 11.0, "lon": 75.0},
    "Hilly": {"lat": 12.5, "lon": 77.5},
    "Plains": {"lat": 20.0, "lon": 78.0},
    "Plateau": {"lat": 22.5, "lon": 85.0},
    "Desert": {"lat": 27.0, "lon": 70.0},
    "River Basin": {"lat": 23.0, "lon": 81.0},
    "Forest": {"lat": 18.5, "lon": 73.5},
    "Marsh": {"lat": 21.5, "lon": 72.5},
    "Valley": {"lat": 30.5, "lon": 79.0},
    "Urban": {"lat": 28.6, "lon": 77.2},
    "Volcanic": {"lat": 19.4, "lon": 72.8},
    "Tropical": {"lat": 8.5, "lon": 77.0},
    "Savanna": {"lat": 14.0, "lon": 75.0},
    "Steppe": {"lat": 34.0, "lon": 75.0},
    "Mangrove": {"lat": 21.5, "lon": 87.0}
}

API_KEY = "1b02228e8e26d606b4d0b40b350e9c0e"

soil_options = ["Sandy","Clay","Loam","Silty","Peaty","Chalky"]
terrain_options = list(terrain_coords.keys())

# Mapping for ML model
soil_map = {name: idx for idx, name in enumerate(soil_options)}
terrain_map = {name: idx for idx, name in enumerate(terrain_options)}

@app.route("/")
def home():
    return render_template("index.html", soil_options=soil_options, terrain_options=terrain_options, input_data=None, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        soil = request.form["soil"]
        terrain = request.form["terrain"]
        rainfall = float(request.form["rainfall"])
        temperature = round(float(request.form["temperature"]))
        humidity = float(request.form["humidity"])

        X = np.array([[soil_map[soil], rainfall, round(temperature), humidity, terrain_map[terrain]]])
        crop_pred_encoded = crop_model.predict(X)[0]
        crop_pred = crop_le.inverse_transform([crop_pred_encoded])[0]

        irrigation_mm = irrigation_model.predict(X)[0]
        if irrigation_mm < 200:
            irrigation_bucket = "Low"
        elif irrigation_mm < 400:
            irrigation_bucket = "Medium"
        else:
            irrigation_bucket = "High"

        result = {
            "crop": crop_pred,
            "irrigation_mm": round(irrigation_mm,1),
            "irrigation_bucket": irrigation_bucket,
            "low_confidence": False,
            "explanation": f"Prediction based on {soil} soil and {terrain} terrain."
        }

        return render_template("index.html", soil_options=soil_options, terrain_options=terrain_options, input_data=request.form, result=result)

    except Exception as e:
        flash(f"Error in prediction: {e}", "danger")
        return render_template("index.html", soil_options=soil_options, terrain_options=terrain_options, input_data=None, result=None)

@app.route("/get_weather_for_terrain")
def get_weather_for_terrain():
    terrain = request.args.get("terrain")
    if terrain not in terrain_coords:
        return jsonify({"error": "Invalid terrain"})

    lat = terrain_coords[terrain]["lat"]
    lon = terrain_coords[terrain]["lon"]
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    try:
        res = requests.get(url).json()
        temperature = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        rainfall = res.get("rain", {}).get("1h", 0)
        return jsonify({"temperature": temperature, "humidity": humidity, "rainfall": rainfall})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
