# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import os
import uvicorn
from feature_extraction import extract_name_features, hash_features
from fastapi.middleware.cors import CORSMiddleware



# Initialize FastAPI app
app = FastAPI(
    title="Gender Prediction API",
    description="API to predict gender from names using machine learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rim-gender-predictor.deno.dev"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and label encoder
model_path = "models/gender_prediction_model/gender_prediction_model/model.pkl"
encoder_path = "models/gender_prediction_model/gender_prediction_model/label_encoder.pkl"

# Check if model files exist
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError(
        "Model files not found. Make sure the model is trained and saved in the gender_prediction_model directory."
    )

# Load model and encoder
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Define Pydantic models for request and response
class NameRequest(BaseModel):
    name: str = Field(..., example="Ahmed", description="The name to predict gender for")

class BatchNameRequest(BaseModel):
    names: List[str] = Field(..., example=["Ahmed", "Fatima", "Mohammed"], 
                           description="List of names to predict gender for")

class GenderPrediction(BaseModel):
    name: str
    predicted_gender: str
    probability: float
    
class PredictionResponse(BaseModel):
    prediction: GenderPrediction
    success: bool = True
    
class BatchPredictionResponse(BaseModel):
    predictions: List[GenderPrediction]
    success: bool = True
    
class ErrorResponse(BaseModel):
    error: str
    success: bool = False

# Define prediction function
def predict_gender(name: str) -> Dict[str, Any]:
    """
    Predict gender for a given name using the loaded model
    """
    try:
        # Extract features
        features = extract_name_features(name)
        
        # Hash features to create feature vector
        feature_vector = hash_features(features)
        
        # Make prediction
        pred = model.predict([feature_vector])[0]
        pred_proba = model.predict_proba([feature_vector])[0]
        
        # Get predicted gender label
        gender = label_encoder.inverse_transform([pred])[0]
        
        # Get probability for the predicted class
        probability = pred_proba.max()
        
        return {
            "name": name,
            "predicted_gender": gender,
            "probability": float(probability)  # Convert numpy float to Python float
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Define API endpoints
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - provides basic API information
    """
    return {
        "message": "Gender Prediction API is running",
        "documentation": "/docs",
        "endpoints": ["/predict", "/predict-batch"]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], 
          responses={500: {"model": ErrorResponse}})
async def predict_single(request: NameRequest):
    """
    Predict gender for a single name
    """
    try:
        prediction = predict_gender(request.name)
        return {"prediction": prediction, "success": True}
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        # Convert other exceptions to HTTP exceptions
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"],
          responses={500: {"model": ErrorResponse}})
async def predict_batch(request: BatchNameRequest):
    """
    Predict gender for multiple names in batch
    """
    try:
        predictions = [predict_gender(name) for name in request.names]
        return {"predictions": predictions, "success": True}
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc 
    except Exception as e:
        # Convert other exceptions to HTTP exceptions
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/model-info", tags=["Model"])
async def model_info():
    """
    Get information about the model
    """
    return {
        "model_type": type(model).__name__,
        "labels": label_encoder.classes_.tolist(),
        "feature_size": model.coef_.shape[1] if hasattr(model, 'coef_') else None
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)