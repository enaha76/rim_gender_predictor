import requests
import json
import time

# Base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the Gender Prediction API endpoints"""
    print("Testing Gender Prediction API...")
    
    # Test names
    test_names = [
        "Ahmed",
        "Fatima",
        "Mohammed",
        "Aisha",
        "Omar",
        "Mariam",
        "Abdallah",
        "Khadija",
        "Ibrahim",
        "Zeinab"
    ]
    
    # Test the prediction endpoint
    print("\n1. Testing prediction endpoint:")
    for name in test_names:
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"name": name}
            )
            
            if response.status_code == 200:
                result = response.json()
                gender = "Male" if result["gender"] == "M" else "Female"
                probability = round(result["probability"] * 100, 2)
                print(f"  {name}: {gender} ({probability}% confidence)")
            else:
                print(f"  {name}: Error - {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"  {name}: Exception - {str(e)}")
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Test the statistics endpoint
    print("\n2. Testing statistics endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"  Total predictions: {stats['total_predictions']}")
            print(f"  Gender distribution: {stats['gender_counts']['male']} male, {stats['gender_counts']['female']} female")
            print(f"  Average confidence: {round(stats['average_confidence']['male'] * 100, 2)}% for male, {round(stats['average_confidence']['female'] * 100, 2)}% for female")
        else:
            print(f"  Error - {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"  Exception - {str(e)}")
    
    # Test the model info endpoint
    print("\n3. Testing model info endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        
        if response.status_code == 200:
            info = response.json()
            print(f"  Model type: {info['model_type']}")
            print(f"  Features: {len(info['features']['types'])} feature types")
            print(f"  Parameters: regularization={info['parameters']['regularization']}, elasticNet={info['parameters']['elasticNet']}")
        else:
            print(f"  Error - {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"  Exception - {str(e)}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    test_api()