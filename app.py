import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("Model/house_price_model.pkl", "rb"))


# Define feature names
feature_columns = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income", "ocean_proximity"
]

# Define categories for ocean_proximity
ocean_categories = {"NEAR BAY": 0, "INLAND": 1, "NEAR OCEAN": 2, "ISLAND": 3, "<1H OCEAN": 4}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.content_type == "application/json":
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Convert inputs to correct data types
        longitude = float(data["longitude"])
        latitude = float(data["latitude"])
        housing_median_age = float(data["housing_median_age"])
        total_rooms = float(data["total_rooms"])
        total_bedrooms = float(data["total_bedrooms"])
        population = float(data["population"])
        households = float(data["households"])
        median_income = float(data["median_income"])
        
        # Get categorical input
        ocean_proximity = data["ocean_proximity"]
        ocean_category = ocean_categories.get(ocean_proximity, 0)

        # Convert input into Pandas DataFrame
        input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms,
                                    total_bedrooms, population, households, median_income, ocean_category]],
                                  columns=feature_columns)

        # Predict house price
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        if request.content_type == "application/json":
            return jsonify({"predicted_price": output})
        else:
            return render_template("index.html", prediction_text=f"Predicted House Price: ${output}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Use Render’s PORT
    app.run(host="0.0.0.0", port=port, debug=False)
