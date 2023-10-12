import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def handle_missing_values(data):
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical column
            mode_val = data[column].mode()[0]
            data[column].fillna(mode_val, inplace=True)
        else:  # Numeric column
            mean_val = data[column].mean()
            data[column].fillna(mean_val, inplace=True)
    return data

def remove_missing_values(data):
    return data.dropna()

def remove_duplicates(data):
    return data.drop_duplicates()

def handle_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the data within the bounds
    for column in data.columns:
        if data[column].dtype != 'object':  # Apply only on numeric columns
            data[column] = data[column].clip(lower_bound[column], upper_bound[column])
    return data

def remove_irrelevant_columns(data):
    # Identify and list constant features
    constant_features = [
        feat for feat in data.columns if data[feat].nunique() == 1
    ]
    data = data.drop(constant_features, axis=1)

    numerical_data = data.select_dtypes(include=[np.number]).iloc[:,0:7]
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.show()

    return data
