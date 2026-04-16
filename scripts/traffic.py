import time
import random
import requests
import os

# Retrieve API URL from environment variables or default to the Docker service name
API_URL = os.getenv("API_URL", "http://fastapi:8000/predict")

def generate_fake_data():
    """Generates randomized data matching the model features."""
    return {
        "age": random.randint(18, 65),
        "gender": random.choice(["Male", "Female"]),
        "tenure": random.randint(1, 24),
        "usage_frequency": random.randint(1, 10),
        "support_calls": random.randint(0, 5),
        "payment_delay": random.randint(0, 10),
        "subscription_type": random.choice(["Basic", "Standard", "Premium"]),
        "contract_length": random.choice(["Monthly", "Quarterly", "Annual"]),
        "total_spend": round(random.uniform(50, 500), 2),
        "last_interaction": random.randint(1, 30)
    }

def main():
    print(f"Starting traffic simulation targeting: {API_URL}")
    while True:
        try:
            payload = generate_fake_data()
            response = requests.post(API_URL, json=payload, timeout=5)
            
            if response.status_code == 200:
                print(f"Success: Prediction received {response.json()}")
            else:
                print(f"Warning: Received status code {response.status_code}")
                
        except Exception as e:
            print(f"Error connecting to API: {e}")
        
        # Adjust frequency of requests (0.5 to 2 seconds)
        time.sleep(random.uniform(0.5, 2.0))

if __name__ == "__main__":
    main()