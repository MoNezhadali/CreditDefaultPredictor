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

def save_data(data, output_path):
    data.to_csv(output_path, index=False)