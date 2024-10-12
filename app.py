# Importing essential libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        data = request.json
        age = int(data['age'])
        sex = int(data['sex'])
        cp = int(data['cp'])
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = int(data['fbs'])
        restecg = int(data['restecg'])
        thalach = int(data['thalach'])
        exang = int(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])

        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction = model.predict(data)

        return jsonify({'prediction': int(my_prediction)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
