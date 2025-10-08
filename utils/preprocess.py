"""
preprocess.py
Helper functions to preprocess and validate inputs
for Crop & Irrigation Prediction models.
"""

import numpy as np

# -------------------------
# Mapping for categorical features
# -------------------------
SOIL_OPTIONS = ["Sandy", "Clay", "Loam", "Silty", "Peaty", "Chalky"]
TERRAIN_OPTIONS = ["Coastal", "Hilly", "Plain"]

SOIL_MAP = {name: idx for idx, name in enumerate(SOIL_OPTIONS)}
TERRAIN_MAP = {name: idx for idx, name in enumerate(TERRAIN_OPTIONS)}

# -------------------------
# Input Validation
# -------------------------
def validate_inputs(soil, rainfall, temperature, humidity, terrain):
    errors = []

    # Required fields
    if soil not in SOIL_MAP:
        errors.append(f"Invalid soil type: {soil}")
    if terrain not in TERRAIN_MAP:
        errors.append(f"Invalid terrain: {terrain}")

    # Numeric checks
    try:
        rainfall = float(rainfall)
        if rainfall < 0:
            errors.append("Rainfall cannot be negative")
    except Exception:
        errors.append("Rainfall must be a number")

    try:
        temperature = float(temperature)
        if not (-50 <= temperature <= 60):
            errors.append("Temperature out of realistic range (-50°C to 60°C)")
    except Exception:
        errors.append("Temperature must be a number")

    try:
        humidity = float(humidity)
        if not (0 <= humidity <= 100):
            errors.append("Humidity must be between 0 and 100")
    except Exception:
        errors.append("Humidity must be a number")

    return errors

# -------------------------
# Feature Conversion
# -------------------------
def convert_to_features(soil, rainfall, temperature, humidity, terrain):
    """
    Converts user-friendly inputs into numeric features for ML model
    Order: [soil_idx, rainfall, temperature, humidity, terrain_idx]
    """
    soil_idx = SOIL_MAP.get(soil, -1)
    terrain_idx = TERRAIN_MAP.get(terrain, -1)

    X = np.array([[soil_idx, float(rainfall), float(temperature),
                   float(humidity), terrain_idx]])

    return X

# -------------------------
# Irrigation bucket conversion
# -------------------------
IRRIGATION_THRESHOLDS_MM = {
    "Low": 200,
    "Medium": 400
    # >400 -> High
}

def mm_to_bucket(mm):
    """Convert irrigation mm to Low/Medium/High using thresholds."""
    if mm <= IRRIGATION_THRESHOLDS_MM["Low"]:
        return "Low"
    elif mm <= IRRIGATION_THRESHOLDS_MM["Medium"]:
        return "Medium"
    else:
        return "High"

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    soil = "Loam"
    rainfall = 1200
    temperature = 26.5
    humidity = 78
    terrain = "Coastal"

    errs = validate_inputs(soil, rainfall, temperature, humidity, terrain)
    if errs:
        print("Validation Errors:", errs)
    else:
        X = convert_to_features(soil, rainfall, temperature, humidity, terrain)
        print("Feature Array:", X)
        print("Irrigation bucket for 350 mm:", mm_to_bucket(350))
