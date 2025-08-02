from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from model import build_model

app = FastAPI()

# Allow all origins (for frontend use if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = build_model()
model.load_weights("goat_weight_model_mobilenet_segmented.h5")

@app.post("/predict-bmi")
async def predict_bmi(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    bmi_value = float(prediction[0][0])
    return {"bmi": bmi_value}
