import io
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from gradcam import get_img_array, make_gradcam_heatmap, overlay_gradcam
from generate_report import generate_medical_report

app = Flask(__name__)
# Enable CORS so the React frontend can make requests
CORS(app)

MODEL_PATH = 'pneumonia_model.h5'
model = None

# We can load the model at startup or upon the first request.
# Loading at startup is better for production.
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Inference will fail until model is trained.")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image_bytes):
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 224x224 as required by the model
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure pneumonia_model.h5 exists and restart the API."}), 503

    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({"error": "No image uploaded. Please send a 'file' or 'image' field."}), 400
    
    file = request.files.get('file') or request.files.get('image')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess
        processed_img = preprocess_image(image_bytes)
        
        # Run inference
        prediction_prob = model.predict(processed_img)[0][0]
        
        # Determine class (assuming 0 = NORMAL, 1 = PNEUMONIA)
        # Based on how flow_from_directory typically assigns classes alphabetically
        is_pneumonia = prediction_prob > 0.5
        
        prediction_label = "PNEUMONIA" if is_pneumonia else "NORMAL"
        
        # Confidence score (probability of the predicted class)
        confidence = float(prediction_prob) if is_pneumonia else float(1.0 - prediction_prob)
        
        # Generate Grad-CAM Heatmap
        # MobileNetV2 last conv layer name
        last_conv_layer_name = "out_relu"
        
        heatmap_base64 = None
        try:
            # We predict using the preprocessed image to get Grad-CAM
            # The gradcam function expects the image to already be an array of shape (1, 224, 224, 3)
            heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name)
            heatmap_base64 = overlay_gradcam(processed_img, heatmap)
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            
        # Generate the medical report
        # We can extract a patient ID if passed in the request, otherwise generate a placeholder
        patient_id = request.form.get('patientId', f"PT-{np.random.randint(10000, 99999)}")
        report_text = generate_medical_report(prediction_label, confidence, patient_id=patient_id)
            
        response = {
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "heatmap": heatmap_base64,
            "report": report_text
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
