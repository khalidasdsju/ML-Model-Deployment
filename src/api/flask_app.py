from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__,
            static_folder='templates/static',
            template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/model-info')
def model_info():
    # Mock data for model info
    return jsonify({
        "model_loaded": True,
        "model_version": "local",
        "model_features": ["FS", "DT", "NYHA", "HR", "BNP", "LVIDs", "BMI", "LAV",
                          "Wall_Subendocardial", "LDLc", "Age", "ECG_T_inversion",
                          "ICT", "RBS", "EA", "Chest_pain"],
        "threshold": 0.5,
        "important_features": ["BNP", "NYHA", "FS", "Age", "HR"]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    # Mock prediction endpoint
    patient_data = request.json

    # Simple mock prediction logic
    if patient_data.get('BNP', 0) > 400 or patient_data.get('NYHA', 0) > 2:
        prediction = 1
        probability = 0.85
    else:
        prediction = 0
        probability = 0.15

    return jsonify({
        "probability": probability,
        "prediction": prediction,
        "threshold": 0.5,
        "model_version": "local",
        "timestamp": "2023-04-09 12:00:00",
        "features_used": ["FS", "DT", "NYHA", "HR", "BNP"],
        "patient_data": patient_data
    })

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    # Mock batch prediction endpoint
    data = request.json
    patients = data.get('patients', [])

    predictions = []
    for patient in patients:
        if patient.get('BNP', 0) > 400 or patient.get('NYHA', 0) > 2:
            prediction = 1
            probability = 0.85
        else:
            prediction = 0
            probability = 0.15

        predictions.append({
            "probability": probability,
            "prediction": prediction,
            "threshold": 0.5,
            "model_version": "local",
            "timestamp": "2023-04-09 12:00:00",
            "features_used": ["FS", "DT", "NYHA", "HR", "BNP"],
            "patient_data": patient
        })

    # Calculate summary
    total = len(predictions)
    positive = sum(1 for p in predictions if p["prediction"] == 1)
    negative = total - positive

    return jsonify({
        "predictions": predictions,
        "summary": {
            "total_patients": total,
            "positive_predictions": positive,
            "negative_predictions": negative,
            "positive_percentage": (positive / total) * 100 if total > 0 else 0,
            "negative_percentage": (negative / total) * 100 if total > 0 else 0
        }
    })

if __name__ == '__main__':
    port = 1020  # Using your preferred port
    app.run(host='0.0.0.0', port=port, debug=True)
