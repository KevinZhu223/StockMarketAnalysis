# test_pipeline.py
import os
from dotenv import load_dotenv
from scripts.data.data_collectionction import collect_recent_data
from scripts.data.data_cleaning import clean_stock_data, clean_news_data
import pandas as pd

# Test with a single ticker
test_ticker = 'AAPL'
days_back = 5  # Last 5 days

# Load environment variables
load_dotenv()

# Create test directories
os.makedirs('data/test/realtime', exist_ok=True)
os.makedirs('data/test/processed', exist_ok=True)

# Test data collection
print(f"Testing real-time data collection for {test_ticker}...")
_, _, stock_path, news_path = collect_recent_data(test_ticker, days_back)

# Test data cleaning
print(f"Testing data cleaning for real-time data...")
stock_clean_path = f"data/test/processed/{test_ticker}_stock_clean.csv"
news_clean_path = f"data/test/processed/{test_ticker}_news_clean.csv"

clean_stock_data(stock_path, stock_clean_path, test_ticker)
clean_news_data(news_path, news_clean_path, days_back)

print("Real-time test completed!")

# Load and display sample of processed data
print("\nSample of processed stock data:")
try:
    stock_df = pd.read_csv(stock_clean_path, index_col=0, parse_dates=True)
    print(stock_df.head())
    print(f"Stock data shape: {stock_df.shape}")
except Exception as e:
    print(f"Error loading stock data: {str(e)}")

print("\nSample of processed news data:")
try:
    news_df = pd.read_csv(news_clean_path, index_col=0, parse_dates=True)
    print(news_df.head())
    print(f"News data shape: {news_df.shape}")
except Exception as e:
    print(f"Error loading news data: {str(e)}")