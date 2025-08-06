
    
# app.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("student_mark_predictor.pkl")

# Initialize empty DataFrame to store inputs and predictions
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df

    try:
        # Get input value from form
        study_hours = int(request.form['study_hours'])

        # Validate input
        if study_hours < 1 or study_hours > 24:
            return render_template('index.html',
                                   prediction_text='⚠️ Please enter a valid number between 1 and 24.')

        # Make prediction
        prediction = model.predict([[study_hours]])[0][0].round(2)

        # Save the result to CSV
        df = pd.concat([df, pd.DataFrame({'Study Hours': [study_hours], 'Predicted Marks': [prediction]})],
                       ignore_index=True)
        df.to_csv('smp_data_from_app.csv', index=False)

        # Show result
        return render_template('index.html',
                               prediction_text=f'✅ You will get approximately {prediction}% marks by studying {study_hours} hours/day.')

    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
