import pandas as pd
import os


#example input file 'data/raw_data/AAPL_raw_data.csv'
#example output file 'data/clean_data/AAPL_clean_data.csv'

def clean_data(input_file, output_file):
    
    df = pd.read_csv(input_file, index_col = 0)

    #print("Raw Data:")
    #print(df)  
    #print(df.isna().sum())

    numeric_columns = ['High','Low', 'Close', 'Open', 'Volume']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    df.dropna(subset=numeric_columns, inplace = True)

    #print(df.describe())

    #df.index = pd.to_datetime(df.index)

    #print("\nCleaned Data: ")
    #print(df.sample(25))
    #print(df.isna().sum())

    os.makedirs(os.path.dirname(output_file), exist_ok = True)
    df.to_csv(output_file)

