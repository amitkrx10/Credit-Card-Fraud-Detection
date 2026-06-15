import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("creditcard.csv")

# Take one sample transaction (random)
sample = df.sample(1)

# Keep target aside
sample_features = sample.drop("Class", axis=1)

# Scale Time and Amount properly
sample_features["Time"] = scaler.transform(sample_features["Time"].values.reshape(-1, 1))
sample_features["Amount"] = scaler.transform(sample_features["Amount"].values.reshape(-1, 1))

# Convert to numpy array
input_data = np.array(sample_features).reshape(1, -1)

# Predict
prediction = model.predict(input_data)

# Output result
if prediction[0] == 1:
    print("\n⚠ Fraud Transaction Detected!")
else:
    print("\n✔ Normal Transaction")