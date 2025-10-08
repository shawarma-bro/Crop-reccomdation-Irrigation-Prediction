"""
trainmodels.py
Sample script to train and save ML models for:
- Crop prediction (classifier)
- Irrigation prediction (regressor)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 1. Load dataset
# -------------------------
# Replace this with your real dataset
# Columns: soil, rainfall, temperature, humidity, terrain, crop, irrigation_mm
#data = pd.read_csv("data/dataset.csv")

# Example columns if you don’t have a dataset yet:
data = pd.DataFrame({
     "soil": np.random.choice(["Sandy","Clay","Loam","Silty","Peaty","Chalky"], 100),
     "rainfall": np.random.randint(100, 2000, 100),
     "temperature": np.random.uniform(15, 35, 100),
     "humidity": np.random.uniform(40, 90, 100),
     "terrain": np.random.choice(["Coastal","Hilly","Plain"], 100),
     "crop": np.random.choice(["Wheat","Rice","Maize","Millet"], 100),
     "irrigation_mm": np.random.randint(100, 600, 100)
 })

# -------------------------
# 2. Encode categorical features
# -------------------------
soil_map = {name: idx for idx, name in enumerate(["Sandy","Clay","Loam","Silty","Peaty","Chalky"])}
terrain_map = {name: idx for idx, name in enumerate(["Coastal","Hilly","Plain"])}

data["soil_idx"] = data["soil"].map(soil_map)
data["terrain_idx"] = data["terrain"].map(terrain_map)

# -------------------------
# 3. Crop Prediction Model
# -------------------------
X_crop = data[["soil_idx","rainfall","temperature","humidity","terrain_idx"]]
y_crop = data["crop"]

crop_le = LabelEncoder()
y_crop_encoded = crop_le.fit_transform(y_crop)

X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop_encoded, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

print(f"Crop model accuracy: {crop_model.score(X_test, y_test):.2f}")

# Save model and encoder
with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

with open("models/label_encoder_crop.pkl", "wb") as f:
    pickle.dump(crop_le, f)

# -------------------------
# 4. Irrigation Prediction Model
# -------------------------
X_irri = data[["soil_idx","rainfall","temperature","humidity","terrain_idx"]]
y_irri = data["irrigation_mm"]

X_train, X_test, y_train, y_test = train_test_split(X_irri, y_irri, test_size=0.2, random_state=42)

irrigation_model = RandomForestRegressor(n_estimators=100, random_state=42)
irrigation_model.fit(X_train, y_train)

print(f"Irrigation model R^2 score: {irrigation_model.score(X_test, y_test):.2f}")

# Save irrigation model
with open("models/irrigation_model.pkl", "wb") as f:
    pickle.dump(irrigation_model, f)

print("✅ Models trained and saved successfully in models/ folder")
