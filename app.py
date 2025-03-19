from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
price_model = joblib.load("property_price_model.pkl")
county_model = joblib.load("county_recommendation_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON input
    
    # Extract user input
    year = data.get("year")
    month = data.get("month")

    # Make prediction
    prediction = price_model.predict([[year, month]])[0]

    return jsonify({"predicted_price": round(prediction, 2)})


@app.route("/recommend", methods=["POST"])
def recommend_county():
    data = request.json
    # Extract user input
    year = data.get("year")
    month = data.get("month")
    budget = data.get("budget")
    
    # Make price prediction
    prediction = price_model.predict([[year, month]])[0]

    # Predict the best county based on affordability
    county_prediction = county_model.predict([[year, month, budget]])[0]
    
    return jsonify({"recommended_county": county_prediction})

if __name__ == "__main__":
    app.run(debug=True)
