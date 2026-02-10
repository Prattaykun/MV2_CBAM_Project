import requests
import sys
import os

def test_prediction(image_path, url="http://localhost:8000/predict"):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return

    print(f"Testing API with image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the backend is running on http://localhost:8000")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to a random image from dataset if available
        # This path might need adjustment based on where you run it from
        img_path = "../dataset/Test/Fire/resized_frame12410.jpg" 
        
    test_prediction(img_path)
