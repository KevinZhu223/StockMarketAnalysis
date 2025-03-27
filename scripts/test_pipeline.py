# Create a test script (test_pipeline.py)
import os
from dotenv import load_dotenv
from scripts.data_collection import collect_stock_data, collect_news_data
from scripts.data_cleaning import clean_stock_data, clean_news_data

# Test with a single ticker and short date range
test_ticker = 'AAPL'
test_start_date = '2023-01-01'
test_end_date = '2023-01-31'  # Just one month of data
load_dotenv()
    
news_api_key = os.getenv("NEWS_API_KEY")    
    
if not news_api_key:
    raise ValueError("API key not found! Make sure it's set in the .env file.")

# Create test directories
os.makedirs('data/test/raw', exist_ok=True)
os.makedirs('data/test/processed', exist_ok=True)

# Test data collection
print("Testing data collection...")
stock_path = f"data/test/raw/{test_ticker}_stock.csv"
news_path = f"data/test/raw/{test_ticker}_news.json"

collect_stock_data(test_ticker, test_start_date, test_end_date, save_path=stock_path)
collect_news_data(test_ticker, test_start_date, test_end_date, news_api_key, save_path=news_path)

# Test data cleaning
print("Testing data cleaning...")
stock_clean_path = f"data/test/processed/{test_ticker}_stock_clean.csv"
news_clean_path = f"data/test/processed/{test_ticker}_news_clean.csv"

clean_stock_data(stock_path, stock_clean_path)
clean_news_data(news_path, news_clean_path)

print("Test completed!")