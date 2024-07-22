# Python script containing functions for drawing different plots and visualizations.

# Import required Python libraries for the analysis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import scipy
from scipy.stats import chi2_contingency
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def plot_count_percentage_missing(data, column_name):
    """
    Function to plot a countplot with count, percentage of categories of a column and missing values in the data
    
    Args:
        data: dataset name
        column_name: name of the column to be plotted.

    Returns: None(Plot of count, percentage and missing values)
    
    """
    # Calculate value counts and percentages
    value_counts = data[column_name].value_counts()
    percentages = (value_counts / len(data[column_name])) * 100

    # Calculate missing values percentage
    missing_percentage = (data[column_name].isnull().sum() / len(data[column_name])) * 100

    # Create a countplot
    # Define custom color palette
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sns.set(style="darkgrid", palette=custom_palette)
    plt.figure(figsize=(10, 4))
    ax = sns.countplot(data=data, x=column_name)

    # Add count, percentage, and missing values percentage labels to the bars
    total = len(data[column_name])
    for p in ax.patches:
        count = int(p.get_height())
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        label = f'{count} ({percentage})'
        x = p.get_x() + p.get_width() / 2 - 0.1
        y = p.get_height() + 0.5
        ax.annotate(label, (x, y), fontsize=10, ha='center')

    # Add missing values percentage to the title
    title = f'Count and Percentage of {column_name.capitalize()} (Missing Data: {missing_percentage:.3f}%)'
    plt.xlabel(column_name.capitalize(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    plt.show()


def describe_numeric_column(df, column_name):
    """
    Function to describe a numeric column in a pandas DataFrame and show missing values.

    Args:
    - df (pd.DataFrame): The pandas DataFrame containing the column.
    - column_name (str): The name of the numeric column to describe.

    Returns:
    - None (prints description and missing values information).
    """
    # Describe the numeric column
    description = df[column_name].describe()
    print(f"Description of '{column_name}':\n{description}\n")
    
    # Check for missing values
    missing_values = df[column_name].isnull().sum()
    total_values = df.shape[0]
    if missing_values > 0:
        print(f"Missing values in '{column_name}': {missing_values} out of {total_values} records")
    else:
        print(f"No missing values found in '{column_name}'")


def visualize_numeric_variables(data, numeric_columns):
    """
    Function to visualize numeric variables in the data with histogram, box plot and violin plot.
    
    Args:
        data: dataset name
        numeric_columns: names of the numeric columns to be plotted.
    
    Returns:
        None(Histogram, Box PLot and Violin PLots of numeric variables.)
    """
    # Set the style for Seaborn plots
    sns.set(style="whitegrid")

    # Loop through each numeric column and create visualizations
    for column in numeric_columns:
        # Create a figure with subplots
        plt.figure(figsize=(12, 6))
        
        # Plot a histogram
        plt.subplot(2, 2, 1)
        sns.histplot(data=data, x=column, kde=True)
        plt.title(f'Histogram of {column}', fontsize=14)

        # Plot a box plot
        plt.subplot(2, 2, 2)
        sns.boxplot(data=data, y=column)
        plt.title(f'Box Plot of {column}', fontsize=14)

        # Plot a violin plot
        plt.subplot(2, 2, 3)
        sns.violinplot(data=data, y=column)
        plt.title(f'Violin Plot of {column}', fontsize=14)

        # Adjust subplot layout
        plt.tight_layout()

        # Show the plots
        plt.show()


def visualize_correlations(data):
    """
    Function to visualize correlations between different columns i.e. heatmap between numeric variables and chi square metric between categorical variables.
    
    Args:
        data: dataset name

    Returns: 
        None(Heatmap of numerical as well as categorical variables)
    
    """
    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(exclude=['number']).columns

    # Calculate and visualize correlations for numeric variables
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        numeric_corr = data[numeric_columns].corr()
        sns.heatmap(numeric_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Variables)', fontsize=16)
        plt.show()

    # Calculate and visualize correlations for categorical variables
    if len(categorical_columns) > 1:
        plt.figure(figsize=(10, 8))
        categorical_corr = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 != col2:
                    crosstab = pd.crosstab(data[col1], data[col2])
                    chi2, _, _, _ = stats.chi2_contingency(crosstab)
                    categorical_corr.at[col1, col2] = chi2

        sns.heatmap(categorical_corr.astype(float), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap (Categorical Variables)', fontsize=16)
        plt.show()


def scatter_plot(data, x_column, y_column):
    """
    Function to plot a scatter plot for a pair of columns provided.
    
    Args:
        data: dataset name
        x_column: name of the column to be plotted on X axis.
        y_column: name of the column to be plotted on Y axis.

    Returns:
        None(Scatter plot of data)
    
    """
    # Create a scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.title(f'Scatter Plot: {x_column} vs. {y_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.show()


def visualize_categorical_impact(data, categorical_columns, target_column='OUTCOME', plot_rows=6, plot_cols=3, figsize=(12, 15), xtick_fontsize=8):
    """
    Function to visualize the impact of each categorical variable on 'Personal Loan' using count plots.

    Args:
    - data (DataFrame): The pandas DataFrame containing the data.
    - categorical_columns (list): List of categorical column names to visualize.
    - target_column (str): Name of the target column (default: 'OUTCOME').
    - plot_rows (int): Number of rows for subplots (default: 3).
    - plot_cols (int): Number of columns for subplots (default: 2).
    - figsize (tuple): Figure size (width, height) in inches (default: (12, 15)).

    Returns:
    - None (displays subplots).
    """
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=figsize)
    
    # Flatten axes if needed
    if plot_rows == 1 and plot_cols == 1:
        axes = np.array([axes])

    for idx, cat_col in enumerate(categorical_columns):
        row, col = idx // plot_cols, idx % plot_cols
        sns.countplot(x=cat_col, data=data, hue=target_column, ax=axes[row, col])
        axes[row, col].set_title(f'Impact of {cat_col} on {target_column}', fontsize=12)
        axes[row, col].set_xlabel(cat_col, fontsize=12)
        axes[row, col].set_ylabel('Count', fontsize=12)
        axes[row, col].legend(title=target_column)
        axes[row, col].tick_params(axis='x', labelsize=xtick_fontsize)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical space between subplots
    plt.show()


def visualize_numeric_impact(data, numerical_columns, target_column='OUTCOME', plot_rows=3, plot_cols=2, figsize=(17, 15)):
    """
    Function to visualize the impact of numeric variables on the target variable using box plots.

    Parameters:
    - data (DataFrame): The pandas DataFrame containing the data.
    - numerical_columns (list): List of numeric column names to visualize.
    - target_column (str): Name of the target column (default: 'OUTCOME').
    - plot_rows (int): Number of rows for subplots (default: 3).
    - plot_cols (int): Number of columns for subplots (default: 2).
    - figsize (tuple): Figure size (width, height) in inches (default: (17, 15)).

    Returns:
    - None (displays subplots).
    """
    if plot_cols == 1 and plot_rows != 1:
        fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(figsize[0], figsize[1] * plot_rows))
        for idx, num_col in enumerate(numerical_columns):
            sns.boxplot(y=num_col, data=data, x=target_column, ax=axes[idx])
            axes[idx].set_title(f'Impact of {num_col} on {target_column}', fontsize=14)
            axes[idx].set_xlabel(target_column, fontsize=12)
            axes[idx].set_ylabel(num_col, fontsize=12)
    else:
        fig, axes = plt.subplots(plot_rows, plot_cols, figsize=figsize)
        if plot_rows == 1 and plot_cols == 1:
            axes = np.array([axes])
        for idx, num_col in enumerate(numerical_columns):
            row, col = idx // plot_cols, idx % plot_cols
            sns.boxplot(y=num_col, data=data, x=target_column, ax=axes[row, col])
            axes[row, col].set_title(f'Impact of {num_col} on {target_column}', fontsize=14)
            axes[row, col].set_xlabel(target_column, fontsize=12)
            axes[row, col].set_ylabel(num_col, fontsize=12)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)  # Adjust vertical space between subplots
    plt.show()
