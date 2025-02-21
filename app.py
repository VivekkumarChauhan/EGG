from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('model_rf.pkl')
scaler = StandardScaler()

# Prediction mapping
prediction_map = {0: 'Healthy', 1: 'Epilepsy', 2: 'Alzheimer\'s'}

@app.route('/')
def home():
    return "Machine Learning API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Extract features from the input
        features = np.array([data['Mean_Amplitude'], data['Peak_to_Peak'], data['PSD'], 
                             data['Entropy'], data['Delta_Power'], data['Theta_Power'], 
                             data['Alpha_Power'], data['Beta_Power'], data['Gamma_Power']]).reshape(1, -1)
        
        # Scale the features using the same scaler as during training
        features_scaled = scaler.fit_transform(features)
        
        # Predict using the trained model
        prediction = model.predict(features_scaled)
        
        # Map the numeric prediction to the actual condition name
        predicted_condition = prediction_map.get(int(prediction[0]), "Unknown")
        
        # Return the prediction
        return jsonify({'prediction': predicted_condition})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
