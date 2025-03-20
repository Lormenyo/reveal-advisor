from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_and_clean_data

# Load cleaned data
df = load_and_clean_data("PPR-ALL.csv")

# Prepare features and labels
df["Year"] = df["Date of Sale"].dt.year
df["Month"] = df["Date of Sale"].dt.month

X = df[["Year", "Month", "Price (€)"]]
y = df["County"]  # We predict the best county based on affordability

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save compressed model (gzip compression)
joblib.dump(classifier, "county_recommendation_model.pkl", compress=3)

print("Classification model trained and saved successfully!")
