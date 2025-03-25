#Downloads a single stock

import yfinance as yf
import pandas as pd


def download_stock_data(stock_sym, start_date, end_date, save_path):
    
    stock_data = yf.download(stock_sym, start = '2020-01-01', end = '2025-01-01')

    stock_data.to_csv(f'data/raw_data/{stock_sym}_raw_data.csv')

#print(data.head())

