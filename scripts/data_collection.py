#Downloads a single stock
import os
import requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import json
from dotenv import load_dotenv



def collect_stock_data(ticker, start_date, end_date, save_path = None):
    print(f"Collecting stock data for {ticker} from {start_date} to {end_date}")
    
    data = yf.download(ticker, start = start_date, end = end_date)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        data.to_csv(save_path)
        print(f"Saved stock data to {save_path}")
    
    return data

def collect_news_data(ticker, start_date, end_date, api_key, save_path = None):
    print(f"Collecting news data for {ticker} from {start_date} to {end_date}")
    
    #get the company name from ticker
    company_info = yf.Ticker(ticker).info
    company_name = company_info.get('shortName', ticker)
    
    base_url = "https://newsapi.org/v2/everything"
    all_articles = []
    
    #Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    #Collect news data in monthy chunks due to API limits
    current_date = end_dt
    
    while current_date > start_dt:
        batch_end = current_date.strftime('%Y-%m-%d')
        batch_start = max((current_date - timedelta(days= 30)).strftime('%Y-%m-%d'), start_date)
    
        #Preparing API request parameters
        params = {
            'q': f"{company_name} OR {ticker}",
            'from': batch_start,
            'to': batch_end,
            'langauge': 'en',
            'sortBy': 'publishedAt',
            'apiKey': api_key
        }
        
        #make Api request
        response = requests.get(base_url, params = params)
        
        #check if request worked
        if response.status_code == 200:
            result = response.json()
            batch_articles = result.get('articles', [])
            
            for article in batch_articles:
                article['ticker'] = ticker
            
            all_articles.extend(batch_articles)
            print(f"Collected {len(batch_articles)} articles for {ticker} from {batch_start} to {batch_end}")
            
        else:
            print(f"Error fetching news: {response.status_code}")
            print(response.text)
        
        current_date = datetime.strptime(batch_start, '%Y-%m-%d')

    #Save data if path provided 
    if save_path:
        os.makeddirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, 'w') as f:
            json.dump(all_articles, f)
        print(f"Saved news data to {save_path}")
    
    
    return all_articles
        
def collect_data_for_tickers(tickers, start_date, end_date, news_api_key):
    for ticker in tickers:
        stock_path = f"data/raw_data/{ticker}_stock.csv"
        collect_stock_data(ticker, start_date, end_date, save_path= stock_path)
        
        news_path = f"data/raw_data/{ticker}_news.json"
        collect_news_data(ticker, start_date,end_date, news_api_key, save_path = news_path)
        
        print(f"Completed data collection for {ticker}")


#Example usage
if __name__ == "__main__":
    tickers = ['APPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    load_dotenv()
    
    news_api_key = os.getenv("NEWS_API_KEY")    
    
    if not news_api_key:
        raise ValueError("API key not found! Make sure it's set in the .env file.")

    print(f"Using API Key: {news_api_key[:5]}*****")  # Hide most of the key for security
    collect_data_for_tickers(tickers, start_date, end_date, news_api_key)
