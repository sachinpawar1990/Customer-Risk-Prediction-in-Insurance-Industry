# Python script containing utility functions for various tasks.

# Import required Python libraries for the analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV


def preprocess_data(df, rare_categories=None):
    """
    Preprocesses the input dataframe by performing a series of data cleaning and transformation steps.

    Parameters:
    df (pd.DataFrame): The input dataframe containing raw data to be preprocessed.
    rare_categories (dict, optional): A dictionary where keys are column names and values are lists of categories 
                                      to be treated as rare. If not provided, rare categories are determined based 
                                      on the training data.

    Returns:
    pd.DataFrame: The preprocessed dataframe.

    """
    # Change dtype of H1, H2, H3, H4 to category
    df[['H1', 'H2', 'H3', 'H4']] = df[['H1', 'H2', 'H3', 'H4']].astype('object')

    # Handle negative values in AGE, HEIGHT, WEIGHT, BMI
    df.loc[df['MARITAL_STATUS'] == 'UNEMPLOYEED', 'MARITAL_STATUS'] = 'SINGLE'
    df['AGE'] = df['AGE'].apply(lambda x: abs(x) if x < 0 else x)
    df['HEIGHT'] = df['HEIGHT'].apply(lambda x: abs(x) if x < 0 else x)
    df['WEIGHT'] = df['WEIGHT'].apply(lambda x: abs(x) if x < 0 else x)
    df['BMI'] = df['BMI'].apply(lambda x: abs(x) if x < 0 else x)

    # Correct BMI values greater than 50
    obese_cust = df[df['BMI'] > 50]
    obese_cust['BMI2'] = round((obese_cust['WEIGHT'] / (obese_cust['HEIGHT'] / 100)) / (obese_cust['HEIGHT'] / 100)).astype(int)
    df.loc[df['BMI'] > 50, 'BMI'] = obese_cust['BMI2']

    # Handle missing values for all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])  # Fill with mode for categorical columns
        else:
            df[col] = df[col].fillna(df[col].median())  # Fill with median for numerical columns

    df[['H1', 'H2', 'H3', 'H4']] = df[['H1', 'H2', 'H3', 'H4']].apply(lambda x: x.astype('int').astype('object'))
    
    # Handle rare categories for H1, H2, H3, H4 if rare_categories is provided
    if rare_categories:
        columns_to_handle_rare = ['H1', 'H2', 'H3', 'H4']
        for col in columns_to_handle_rare:
            df[col] = df[col].apply(lambda x: 'RARE' if x in rare_categories[col] else x)

    # Define mapping for each column
    gender_map = {'F': 0, 'M': 1}
    yesno_map = {'No': 0, 'Yes': 1}
    outcome_map = {False: 1, True: 0}

    # Replace categorical variables with the mapping mentioned above
    for col, mapping in zip(['GENDER', 'SMOKING', 'DRINKING', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'OUTCOME'], [gender_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, outcome_map]):
        df[col] = df[col].map(mapping)
    
    # Perform one-hot encoding for specified categorical columns
    # List of columns to one-hot encode
    one_hot_cols = ['MARITAL_STATUS', 'H1', 'H2', 'H3', 'H4']

    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(df[one_hot_cols], prefix=one_hot_cols)
    one_hot_encoded = one_hot_encoded.astype(int)

    # Replace original columns with one-hot encoded columns
    df = pd.concat([df.drop(one_hot_cols, axis=1), one_hot_encoded], axis=1)

    # Remove columns with '_RARE' in their names after one-hot encoding
    columns_to_drop = [col for col in df.columns if '_RARE' in col or 'WIDOW' in col]
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Remove specified columns
    columns_to_drop = ['ID', 'HEIGHT', 'WEIGHT', 'RISK', 'BMI2']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')

    return df


def preprocess_data_api(df, rare_categories=None):
    """
    Preprocesses the input dataframe by performing a series of data cleaning and transformation steps for an API Call.

    Parameters:
    df (pd.DataFrame): The input dataframe containing raw data to be preprocessed.
    rare_categories (dict, optional): A dictionary where keys are column names and values are lists of categories 
                                      to be treated as rare. If not provided, rare categories are determined based 
                                      on the training data.

    Returns:
    pd.DataFrame: The preprocessed dataframe.

    """

    # Handle negative values in AGE, HEIGHT, WEIGHT, BMI
    df.loc[df['MARITAL_STATUS'] == 'UNEMPLOYEED', 'MARITAL_STATUS'] = 'SINGLE'
    df['AGE'] = df['AGE'].apply(lambda x: abs(x) if x < 0 else x)
    df['HEIGHT'] = df['HEIGHT'].apply(lambda x: abs(x) if x < 0 else x)
    df['WEIGHT'] = df['WEIGHT'].apply(lambda x: abs(x) if x < 0 else x)
    df['BMI'] = df['BMI'].apply(lambda x: abs(x) if x < 0 else x)

    # Correct BMI values greater than 50
    obese_cust = df[df['BMI'] > 50]
    obese_cust['BMI2'] = round((obese_cust['WEIGHT'] / (obese_cust['HEIGHT'] / 100)) / (obese_cust['HEIGHT'] / 100)).astype(int)
    df.loc[df['BMI'] > 50, 'BMI'] = obese_cust['BMI2']

    # Handle missing values for all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])  # Fill with mode for categorical columns
        else:
            df[col] = df[col].fillna(df[col].median())  # Fill with median for numerical columns


    # Define mapping for each column
    gender_map = {'F': 0, 'M': 1}
    yesno_map = {'No': 0, 'Yes': 1}
    outcome_map = {False: 1, True: 0}

    # Replace categorical variables with the mapping mentioned above
    for col, mapping in zip(['GENDER', 'SMOKING', 'DRINKING', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'OUTCOME'], [gender_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, yesno_map, outcome_map]):
        df[col] = df[col].map(mapping)
    

    # Remove columns with '_RARE' in their names after one-hot encoding
    columns_to_drop = [col for col in df.columns if '_RARE' in col or 'WIDOW' in col]
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Remove specified columns
    columns_to_drop = ['ID', 'HEIGHT', 'WEIGHT', 'RISK', 'BMI2','MARITAL_STATUS', 'H1', 'H2', 'H3', 'H4']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')

    return df


def scale_data(df):
    """
    Function to do scaling on numerical columns in a DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with scaling done on numerical columns.
    """
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Check if 'OUTCOME' column exists before dropping it
    if 'OUTCOME' in numerical_cols:
        numerical_cols = numerical_cols.drop('OUTCOME')
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def remove_skewness(df, skew_threshold=0.5):
    """
    Function to remove skewness from numerical columns in a DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    skew_threshold (float): The skewness threshold above which a transformation will be applied.

    Returns:
    pd.DataFrame: The DataFrame with skewness removed from numerical columns.
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Check if 'OUTCOME' column exists before dropping it
    if 'OUTCOME' in numerical_cols:
        numerical_cols = numerical_cols.drop('OUTCOME')

    for col in numerical_cols:
        # Calculate skewness
        skewness = df[col].skew()

        # Apply log transformation if skewness exceeds the threshold
        if abs(skewness) > skew_threshold:
            df[col] = np.log1p(df[col])

    return df


def evaluate_model_performance(y_true, y_pred, y_pred_proba=None):
    """
    Function to evaluate a model performance by calculating Precision, Recall, F1 Score, Classification Report and Confusion Matrix
    
    Args:
        y_true: True values of test set
        y_pred: model predictions
        y_pred_proba: probabilities of model predictions

    Returns:
        Show the plots and model performance metrics
    
    """
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    

    # Generate a classification report
    class_report = classification_report(y_true, y_pred)

    # Generate a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linewidths=0.5, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted', fontsize=9)
    plt.ylabel('Actual', fontsize=9)
    plt.title('Confusion Matrix', fontsize=9)

    # Display the performance metrics
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print('\nClassification Report:\n', class_report)

    # Show the confusion matrix
    plt.show()

    if y_pred_proba is not None:
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f'ROC AUC: {roc_auc:.3f}')
        
        # Plot ROC Curve
        RocCurveDisplay.from_predictions(y_true, y_pred_proba)
        plt.title('ROC Curve')
        plt.show()

        # Plot Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba)
        plt.title('Precision-Recall Curve')
        plt.show()


def evaluate_and_save_model_performance(y_true, y_pred, y_pred_proba=None):
    """
    Function to evaluate a model performance by calculating Precision, Recall, F1 Score, Classification Report and Confusion Matrix
    
    Args:
        y_true: True values of test set
        y_pred: model predictions
        y_pred_proba: probabilities of model predictions

    Returns:
        Saves the plots and show model performance metrics
    
    """
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    

    # Generate a classification report
    class_report = classification_report(y_true, y_pred)

    # Generate a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linewidths=0.5, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted', fontsize=9)
    plt.ylabel('Actual', fontsize=9)
    plt.title('Confusion Matrix', fontsize=9)
    plt.savefig('outputs/confusion_matrix.png')

    # Display the performance metrics
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print('\nClassification Report:\n', class_report)

    # Save precision, recall, and F1 score to a text file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f'outputs/performance_metrics_{timestamp}.txt'

    # Save precision, recall, and F1 score to a text file
    with open(metrics_filename, 'w') as f:
        f.write(f'Precision: {precision:.3f}\n')
        f.write(f'Recall: {recall:.3f}\n')
        f.write(f'F1 Score: {f1:.3f}\n')
        f.write('\nClassification Report:\n')
        f.write(class_report)

    # Show the confusion matrix
    plt.show()

    if y_pred_proba is not None:
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f'ROC AUC: {roc_auc:.3f}')
        with open(metrics_filename, 'a') as f:
            f.write(f'ROC AUC: {roc_auc:.3f}\n')
        
        # Plot ROC Curve
        RocCurveDisplay.from_predictions(y_true, y_pred_proba)
        plt.title('ROC Curve')
        plt.savefig('outputs/ROC_Curve.png')
        plt.show()

        # Plot Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba)
        plt.title('Precision-Recall Curve')
        plt.savefig('outputs/Precision_Recall_Display.png')
        plt.show()



def save_model(model, filename):
    """
    Save a trained model to a .pkl file.
    
    Parameters:
    - model: The trained model object (e.g., RandomForestClassifier, LogisticRegression, etc.)
    - filename: Name of the file to save the model (e.g., 'trained_model.pkl')
    
    Returns:
    - None
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")


def get_rare_categories(train_df, columns, rare_threshold=0.002):
    """
    Identify rare categories in categorical columns based on a threshold percentage.

    Parameters:
    train_df (DataFrame): The training DataFrame containing categorical columns.
    columns (list): List of column names to analyze for rare categories.
    rare_threshold (float, optional): Threshold percentage below which categories are considered rare. 
                                      Defaults to 0.002 (0.2%).

    Returns:
    dict: A dictionary where keys are column names and values are lists of rare categories found in each column.
    """
    rare_categories = {}
    for col in columns:
        train_counts = train_df[col].value_counts(normalize=True)
        rare_categories[col] = train_counts[train_counts < rare_threshold].index.tolist()
    return rare_categories


def feature_selection_with_rf(X, y):
    """
    Perform feature selection using RandomForestClassifier as the base estimator.

    Parameters:
    X (DataFrame or array-like of shape (n_samples, n_features)): Input features.
    y (Series or array-like of shape (n_samples,)): Target variable.

    Returns:
    selected_features (array-like of shape (n_selected_features,)): Selected feature names based on importance.
    rf (RandomForestClassifier): Trained RandomForestClassifier used for feature selection.
    """
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features, rf


def plot_feature_importances(importances, feature_names):
    """
    Plot feature importances using horizontal bar chart.

    Parameters:
    importances (array-like of shape (n_features,)): Feature importances from a model.
    feature_names (array-like of shape (n_features,)): Names of features corresponding to importances.

    Returns:
    None
    """
    # Convert importances and feature_names to numpy arrays if they are not already
    importances = np.array(importances)
    feature_names = np.array(feature_names)
    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from Random Forest')
    plt.show()


def train_evaluate_model(model, X, y, test_size=0.2, random_state=42, param_grid=None, feature_selection=False):
    """
    Train and evaluate a machine learning model.

    Parameters:
    model : estimator object
        The machine learning model to be trained and evaluated.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The feature matrix.
    y : array-like of shape (n_samples,)
        The target labels.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    param_grid : dict or None, default=None
        Dictionary with hyperparameters to search over using GridSearchCV. If None, the model is trained with default hyperparameters.
    feature_selection : bool, default=False
        Whether to perform feature selection using RandomForestClassifier with Recursive Feature Elimination (RFE).

    Returns:
    None
    """
    
    if feature_selection:
        # Feature selection
        selected_features, rfe_rf = feature_selection_with_rf(X, y)
        X_selected = X[selected_features]
        X = X_selected

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Cross-validation
    if param_grid:
        random_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='f1', random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    evaluate_model_performance(y_test, y_pred, y_pred_proba)