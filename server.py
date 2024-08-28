from flask import Flask, request, jsonify
from flask_cors import CORS  # Tambahkan impor ini
import typing
import cv2
import numpy as np
import onnxruntime as ort
from mltu.configs import BaseModelConfigs
from mltu.utils.text_utils import ctc_decoder
from mltu.inferenceModel import OnnxInferenceModel

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua rute

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
        
        # Get input shape and input name from the model
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape[1:]  # Assuming NHWC format

    def predict(self, image: np.ndarray):
        # Resize image to match model input shape
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # Expand dimensions to match expected model input
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Run the model inference
        preds = self.model.run(None, {self.input_name: image_pred})[0]

        # Decode the predictions
        text = ctc_decoder(preds, self.char_list)[0]

        return text

# Load konfigurasi model dan inisialisasi model satu kali saat server dimulai
configs = BaseModelConfigs.load("Models/02_captcha_to_text/202408281655/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Convert file to OpenCV image
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Predict using the model
        prediction_text = model.predict(image)

        return jsonify({'prediction': prediction_text})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
