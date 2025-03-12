#Downloads a single stock

import yfinance as yf
import pandas as pd
#First Getting Data for AAPL
stock_sym = 'AAPL'
AAPL_data = yf.download(stock_sym, start = '2020-01-01', end = '2025-01-01')

AAPL_data.to_csv(f'data/raw_data/{stock_sym}_raw_data.csv')

#print(data.head())

