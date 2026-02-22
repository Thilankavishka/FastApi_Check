from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI(title="House Price ML API")

# ===============================
# Step 1: Prepare Data
# ===============================
X = np.array([[1000], [1500], [2000], [2500]])
y = np.array([200000, 300000, 400000, 500000])

# ===============================
# Step 2: Train Model
# ===============================
model = LinearRegression()
model.fit(X, y)


# ===============================
# Step 3: Input Schema
# ===============================
class House(BaseModel):
    size: float


# ===============================
# Step 4: Root Endpoint
# ===============================
@app.get("/")
def home():
    return {"message": "ML Model is running successfully!"}


# ===============================
# Step 5: Prediction Endpoint
# ===============================
@app.post("/predict")
def predict_price(house: House):
    
    # Convert input to numpy array
    input_data = np.array([[house.size]])
    
    # Predict
    prediction = model.predict(input_data)

    return {
        "input_size": house.size,
        "predicted_price": float(prediction[0])
    }