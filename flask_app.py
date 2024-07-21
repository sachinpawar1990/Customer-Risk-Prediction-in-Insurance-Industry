# Python script for a Flask web application, exposing API endpoints.

# Import required Python libraries for the analysis
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from utils import preprocess_data, scale_data, preprocess_data_api
import pickle

app = Flask(__name__)

# Load the model and training data once when the server starts
model = pickle.load(open('Log_reg_model.pkl', 'rb'))
X_train = pickle.load(open('Training_Data.pkl', 'rb'))

rare_categories ={'H1': [4.0, 5.0, 6.0, 7.0, 9.0],
 'H2': [3, 4],
 'H3': [7,8,9,10,11,12,14,13,16,15,17,21,19,25,26,22,27,18,20,37,32,35],
 'H4': [7, 8, 9, 10, 11, 12, 13]}


@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
        # Convert JSON to pandas DataFrame
        df = pd.DataFrame(data['input'])

        # Perform preprocessing on the DataFrame
        df_processed = preprocess_data_api(df, rare_categories= rare_categories)

        # Scale the DataFrame (if necessary)
        df_scaled = scale_data(df_processed)

        # Assuming X_test contains the processed features for prediction
        X_test = df_scaled.drop(columns=['OUTCOME'])  
        y_test = df_scaled["OUTCOME"]

        # Make predictions
        predictions = model.predict(X_test)
        prediction_probabilities = model.predict_proba(X_test)

        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'prediction_probabilities': prediction_probabilities.tolist()
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)