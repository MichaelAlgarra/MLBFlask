# app.py
# This is a flask application that will host a Machine learning model which has been trained and will
# allow for users to acquire information about the outcome of a pitch
# 
# NOTE: alias created - run_mlb_app to run this file
# local env http://127.0.0.1:5000/


from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from model import MockModel

app = Flask(__name__)

# Initialize your model
model = MockModel()

# Render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    playerfirstname = float(request.form.get('playerfirstname'))
    input2 = float(request.form.get('input2'))

    # Prepare input for the model (this depends on your model's input format)
    inputs = np.array([playerfirstname, input2])

    # Run the model
    prediction = model.predict(inputs)

    # Return the result as a JSON response (you could also return a new HTML page)
    return jsonify({'prediction': prediction})

# 
@app.route('/player')
def player():
    return render_template('.html')


if __name__ == '__main__':
    app.run(debug=True)
