# Customer-Risk-Prediction-in-Insurance-Industry
Objective: Predicting the risk associated with customer profiles in the insurance industry.

Description: This project focuses on predicting customer risk in the insurance industry using machine learning models. It includes data exploration, model training, evaluation, batch processing scripts, web application endpoints, Docker configuration, and output visualization.

Motivation: 
Importance of Risk Prediction: Enhancing accuracy in risk assessment for better policy pricing and fraud prevention.

Project Overview:
Dataset: Customer profiles and their corresponding risk ratings.
Approach: Utilizing machine learning models to predict risk and assess model performance.

Files Included
1. Risk_Prediction.ipynb - Jupyter notebook guiding through Data Exploration, Model Training, and Evaluation Process.

2. test_predictions.py - Python script for batch processing of test data.

3. utils.py - Python script containing utility functions for various tasks.

4. common_functions.py - Python script containing functions for drawing different plots and visualizations.

5. training_set.csv - Training dataset used for model training.

6. test_set.csv - New test dataset for demonstrating batch processing.

7. Training_data.pkl - Pickle file containing serialized training data.

8. Log_reg_model.pkl - Pickle file containing a trained logistic regression model.

9. fastapi_app.py - Python script for a FastAPI web application, which starts a uvicorn server and exposes API endpoints.

10. flask_app.py - Python script for a Flask web application, exposing API endpoints.

11. Dockerfile - Dockerfile for building a Docker image based on project specifications.

12. Outputs/Folder containing various output visualizations:
confusion_matrix.png
Precision_Recall_Display.png
ROC_Curve.png
SHAP_summary_plot.png

13. predictions_with_features.csv - CSV file containing original predictors and predictions.

14. requirements.txt - List of Python packages and versions required to set up the environment.


Additional Instructions
1. Batch Processing
Batch processing can be executed using test_predictions.py:
python test_predictions.py


Running the Web Applications

1. FastAPI App
Start the FastAPI web app using:
python fastapi_app.py


Access the API at: http://localhost:8000/predict by passing JSON data.

2. Flask App
Start the Flask web app using:
python flask_app.py

Access the API at: http://localhost:5000/predict by passing JSON data.


Docker
Build a Docker image (my-fastapi-app in this example) using the Dockerfile:
docker build -t my-fastapi-app .

docker build -t my-fastapi-app .

Run the Docker container on port 8000:

docker run -d -p 8000:8000 my-fastapi-app

Once running, access the API at your local system's port mapped to Docker's port (e.g., http://localhost:8000/predict).

