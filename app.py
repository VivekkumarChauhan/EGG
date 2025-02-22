from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('model_rf.pkl')
scaler = StandardScaler()

prediction_map = {0: 'Healthy', 1: 'Epilepsy', 2: 'Alzheimer\'s'}

@app.route('/')
def home():
    return "Machine Learning API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array([data['Mean_Amplitude'], data['Peak_to_Peak'], data['PSD'], 
                             data['Entropy'], data['Delta_Power'], data['Theta_Power'], 
                             data['Alpha_Power'], data['Beta_Power'], data['Gamma_Power']]).reshape(1, -1)
        
        features_scaled = scaler.fit_transform(features)
        prediction = model.predict(features_scaled)
        predicted_condition = prediction_map.get(int(prediction[0]), "Unknown")
        
        return jsonify({'prediction': predicted_condition})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
