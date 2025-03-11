
import joblib
import numpy as np
from feature_extraction import extract_name_features, hash_features

# Load the model and label encoder
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_gender(name):
    """Predict gender for a given name"""
    # Extract features
    features = extract_name_features(name)
    
    # Hash features to create feature vector
    feature_vector = hash_features(features)
    
    # Make prediction
    pred = model.predict([feature_vector])[0]
    pred_proba = model.predict_proba([feature_vector])[0]
    
    # Get predicted gender label
    gender = le.inverse_transform([pred])[0]
    
    # Get probability for the predicted class
    probability = pred_proba.max()
    
    return {
        'name': name,
        'predicted_gender': gender,
        'probability': probability
    }

# Example usage
if __name__ == "__main__":
    # Test with sample names
    test_names = ["Ahmed", "Fatima", "Mohammed", "Aisha"]
    for name in test_names:
        result = predict_gender(name)
        print(f"Name: {result['name']} → Predicted gender: {result['predicted_gender']} (Confidence: {result['probability']:.2f})")
