import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def save_data(data, output_path):
    data.to_csv(output_path, index=False)