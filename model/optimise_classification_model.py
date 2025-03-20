import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load trained model
classifier = joblib.load("county_recommendation_model.pkl")

# Define input shape for conversion
initial_type = [("float_input", FloatTensorType([None, 3]))]

# Convert model to ONNX format
onnx_model = convert_sklearn(classifier, initial_types=initial_type)

# Save ONNX model
with open("county_recommendation_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved successfully!")
