import pandas as pd

from InputOutput import load_data,save_data
from DataCleaningFunctions import remove_missing_values, remove_duplicates

def main():
    input_file = './Data/train.csv'
    output_file = './CleanData/cleaned_training_data.csv'
    # Loading the data
    data = load_data(input_file)
    # Clearning the data
    data = remove_missing_values(data)
    data = remove_duplicates(data)
    # Saving the clean data
    save_data(data, output_file)
    print("Data cleaning and preprocessing completed!")
   

if __name__=="__main__":
    main()