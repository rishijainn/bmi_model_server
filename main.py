from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

from model import build_model

app = FastAPI()

# Allow all origins (for frontend use if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = build_model()
        model_path = "goat_weight_model_mobilenet_segmented.h5"
        
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print("Model weights loaded successfully")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "Goat BMI Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict-bmi")
async def predict_bmi(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).resize((224, 224)).convert("RGB")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        bmi_value = float(prediction[0][0])
        return {"bmi": bmi_value}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)