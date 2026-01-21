from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# إعداد التطبيق
app = FastAPI(title="Sand Inspector API", version="1.0")

# تحميل الموديل مرة واحدة
print("⏳ Loading Model...")
model = tf.keras.models.load_model('model.keras')
print("✅ Model Loaded Successfully!")

@app.get("/")
def home():
    return {"message": "🚀 FastAPI is Running on Hugging Face!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # قراءة الملف
    contents = await file.read()
    
    # معالجة الصورة
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # التوقع
    prediction = model.predict(img_array)[0][0]
    
    prob_unacceptable = float(prediction)
    prob_acceptable = 1.0 - prob_unacceptable
    
    label = "Unacceptable"
    confidence = prob_unacceptable
    
    if prob_acceptable > 0.5:
        label = "Acceptable"
        confidence = prob_acceptable

    return {
        "prediction": label,
        "confidence": f"{confidence*100:.2f}%",
        "details": {
            "acceptable_score": float(prob_acceptable),
            "unacceptable_score": float(prob_unacceptable)
        }
    }
