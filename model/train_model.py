import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # To save the model

# Load data
from data_processing import load_and_clean_data
df = load_and_clean_data("PPR-ALL.csv")

# Feature engineering
df["Year"] = df["Date of Sale"].dt.year
df["Month"] = df["Date of Sale"].dt.month

# Select features & target
X = df[["Year", "Month"]]
y = df["Price (€)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "property_price_model.pkl")

print("Model trained and saved successfully!")
