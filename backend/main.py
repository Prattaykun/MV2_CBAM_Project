import sys
import os
import shutil
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

# Add project root to path to import ml_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_core.predict import FirePredictor

app = FastAPI(title="MV2-CBAM Fire Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
# Ensure the model path is correct based on where train_runner.py saves it
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml_core', 'models', 'mv2_cbam_best.pth')
predictor = FirePredictor(model_path=MODEL_PATH, device='cpu')

@app.get("/")
def root():
    return {"message": "MV2-CBAM Fire Detection API is Running"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Predict
        predicted_class, confidence = predictor.predict(temp_filename)
        
        # Generate Grad-CAM
        cam_image, _ = predictor.generate_cam(temp_filename)
        
        # Encode CAM to base64
        _, buffer = cv2.imencode('.jpg', cam_image)
        cam_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "cam_image_base64": cam_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
