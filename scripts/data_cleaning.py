import pandas as pd
import os

input_file = 'data/raw_data/AAPL_raw_data.csv'

output_file = 'data/clean_data/APPL_clean_data.csv'

df = pd.read_csv(input_file, index_col = 0)

#print("Raw Data:")
#print(df)
#print(df.isna().sum())

df.dropna(inplace = True)

#df.index = pd.to_datetime(df.index)

#print("\nCleaned Data: ")
#print(df.sample(25))
#print(df.isna().sum())

os.makedirs(os.path.dirname(output_file), exist_ok = True)
df.to_csv(output_file)

