from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'soil_classification.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]

        # Predict
        input_array = np.array([features])
        prediction = model.predict(input_array)
        crop = prediction[0]

        # Get prediction probabilities for confidence
        proba = model.predict_proba(input_array)
        confidence = round(float(np.max(proba)) * 100, 1)

        return jsonify({
            'success': True,
            'crop': str(crop),
            'confidence': confidence
        })

    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
