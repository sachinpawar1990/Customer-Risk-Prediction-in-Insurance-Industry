### Python script for batch processing of test data.


# Import required Python libraries for the analysis
import pandas as pd
import pickle
from utils import preprocess_data, scale_data, evaluate_model_performance, evaluate_and_save_model_performance
import shap
import matplotlib.pyplot as plt
import os

# Create the 'outputs' directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)


# Load the saved model
filename = 'Log_reg_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

rare_categories ={'H1': [4.0, 5.0, 6.0, 7.0, 9.0],
 'H2': [3, 4],
 'H3': [7,8,9,10,11,12,14,13,16,15,17,21,19,25,26,22,27,18,20,37,32,35],
 'H4': [7, 8, 9, 10, 11, 12, 13]}

# Assuming test_data.csv contains your new test data
test_data = pd.read_csv('test_set.csv')

# Perform preprocessing on test_data
test_data_processed = preprocess_data(test_data, rare_categories= rare_categories)
test_data_scaled = scale_data(test_data_processed)

# Assuming X_test contains the processed features for prediction
X_test = test_data_scaled.drop(columns=['OUTCOME'])
y_test = test_data_scaled["OUTCOME"]

# Make predictions
predictions = loaded_model.predict(X_test)
y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

evaluate_and_save_model_performance(y_test, predictions, y_pred_proba)

# save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['Predicted_OUTCOME'])
# Concatenate original features with predictions
predictions_with_features = pd.concat([test_data, predictions_df], axis=1)
# Save to CSV
predictions_with_features.to_csv('predictions_with_features.csv', index=False)

# Compute SHAP values
explainer = shap.Explainer(loaded_model, X_test)
shap_values = explainer(X_test)

# Save SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
shap_df.to_csv('shap_values.csv', index=False)

# Plot and save SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type='bar')
plt.title('SHAP Summary Plot')
plt.savefig('outputs/shap_summary_plot.png', bbox_inches='tight')