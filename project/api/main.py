import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io

# تعريف التطبيق
app = Flask(__name__)

# إعدادات الموديل
MODEL_PATH = 'model.keras'
model = None

# دالة تحميل الموديل (تتم مرة واحدة عند بدء التشغيل)
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")

@app.route('/', methods=['GET'])
def index():
    return "🏗️ AI Sand Inspection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    # التأكد من تحميل الموديل
    if model is None:
        load_model()

    # التأكد من وجود ملف في الطلب
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # 1. قراءة الصورة ومعالجتها
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 2. التوقع
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # 3. تفسير النتيجة
        prob_unacceptable = float(prediction)
        prob_acceptable = 1.0 - prob_unacceptable
        
        label = "Unacceptable"
        confidence = prob_unacceptable
        
        if prob_acceptable > 0.5:
            label = "Acceptable"
            confidence = prob_acceptable

        # 4. إرسال الرد (JSON)
        response = {
            'prediction': label,
            'confidence_score': f"{confidence*100:.2f}%",
            'details': {
                'acceptable_prob': f"{prob_acceptable:.4f}",
                'unacceptable_prob': f"{prob_unacceptable:.4f}"
            },
            'engineering_decision': 'SAFE' if prob_acceptable > 0.8 else 'REJECTED/WARNING'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))