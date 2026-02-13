import requests
import os

# Configuration
url = "http://127.0.0.1:5000/predict"
image_path = "metal/metal29.jpg"

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    print("Please make sure the image path is correct.")
    exit(1)

# Test API connection first
try:
    health_check = requests.get("http://127.0.0.1:5000/")
    print(f"API Status: {health_check.json()}")
except Exception as e:
    print(f"Cannot connect to API: {e}")
    print("Make sure the Flask server is running!")
    exit(1)

# Send prediction request
try:
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction successful!")
        print(f"Result: {result}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"Response: {response.json()}")

except Exception as e:
    print(f"Error during request: {e}")