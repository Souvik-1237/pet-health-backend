from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)   # ✅ MUST BE FIRST

model = load_model("model/pet_disease_model.h5")
classes = ["dental_disease", "eye_infection","fungal_infection", "healthy", "hot_spot", "kennal_cough", "mange", "parvovirus" , "skin_allergy", "worm_infection"]

@app.route('/')
def home():
    return "API Running"

@app.route('/predict', methods=['POST'])
def predict():
    # Check form-data
    if 'image' in request.files:
        file = request.files['image']
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

    # Binary method
    else:
        filepath = os.path.join("uploads", "temp.jpg")
        with open(filepath, "wb") as f:
            f.write(request.data)

    # Process image
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    return jsonify({
        "prediction": result,
        "confidence": float(np.max(prediction))
    })

if __name__ == '__main__':
    app.run(debug=True)