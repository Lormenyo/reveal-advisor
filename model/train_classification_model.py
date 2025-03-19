import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import load_and_clean_data

# Load cleaned data
df = load_and_clean_data("PPR-ALL.csv")

# Define affordability threshold
def affordability_label(price, salary):
    return "Affordable" if price <= 4 * salary else "Expensive"

# Example: Assume a fixed average salary per county (Can be dynamic with real salary data)
avg_salary_by_county = {
    "Dublin": 55000, "Cork": 48000, "Galway": 46000, "Limerick": 47000, 
    "Meath": 45000, "Kilkenny": 44000, "Laois": 43000, "Waterford": 44000
}

# Create affordability labels
df["Affordability"] = df.apply(lambda row: affordability_label(row["Price (€)"], avg_salary_by_county.get(row["County"], 40000)), axis=1)

# Prepare features and labels
X = df[["Year", "Month", "Price (€)"]]
y = df["County"]  # We predict the best county based on price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save model
joblib.dump(classifier, "county_recommendation_model.pkl")

print("Classification model trained and saved successfully!")
