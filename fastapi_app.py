# Python script for a FastAPI web application, which starts a uvicorn server and exposes API endpoints.

# Import required Python libraries for the analysis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO
from utils import preprocess_data, scale_data, preprocess_data_api
import pickle

# Load the model and training data once when the server starts
model = pickle.load(open('Risk_Prediction_GB_model.pkl', 'rb'))
X_train = pickle.load(open('Training_Data.pkl', 'rb'))


rare_categories ={'H1': [4.0, 5.0, 6.0, 7.0, 9.0],
 'H2': [3, 4],
 'H3': [7,8,9,10,11,12,14,13,16,15,17,21,19,25,26,22,27,18,20,37,32,35],
 'H4': [7, 8, 9, 10, 11, 12, 13]}

app = FastAPI()

class PredictionInput(BaseModel):
    input_data: dict


@app.post('/predict')
async def predict(input_data: PredictionInput):
    try:
        
        df = pd.DataFrame([input_data.input_data])
        # Perform preprocessing on test_data (apply the same preprocessing steps as used on the training data)
        df_processed = preprocess_data_api(df, rare_categories= rare_categories)
        df_scaled = scale_data(df_processed)

        # Assuming X_test contains the processed features for prediction
        X_test = df_scaled.drop(columns=['OUTCOME'])  # Adjust this based on your preprocessing steps
        y_test = df_scaled["OUTCOME"]
        
        # Make predictions
        predictions = model.predict(X_test)
        prediction_probabilities = model.predict_proba(X_test)
        
        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'prediction_probabilities': prediction_probabilities.tolist()
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
