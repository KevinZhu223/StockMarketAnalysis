#Downloads a single stock
import os
import requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import json
from dotenv import load_dotenv
import numpy as np



def collect_stock_data(ticker, start_date, end_date, save_path = None):
    print(f"Collecting stock data for {ticker} from {start_date} to {end_date}")
    try:
        data = yf.download(ticker, start = start_date, end = end_date, 
                           progress = False, ignore_tz = True, threads = False, auto_adjust=False)
        
        
        if data.empty:
            print(f"Warning: no data retrieved for {ticker}. Trying alternative")
            
             # Try with a different method - using period instead of date range
            try:
                data = yf.download(ticker, period="1mo", 
                                 progress=False, ignore_tz=True, auto_adjust=False)
            except Exception as e:
                print(f"Alternative approach failed: {str(e)}")  
                 
                if not data.empty:
                    print(f"Successfully retrieved data using period parameter")
                
                data = generate_mock_stock_data(ticker, start_date, end_date) 

        if save_path and not data.empty:
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            data.to_csv(save_path)
            print(f"Saved stock data to {save_path}")
    
        return data
    
    except Exception as e:
        print(f"Error collecting stock data for {ticker}: {str(e)}")
        
        print("Generating mock stock data as fallback")
        data = generate_mock_stock_data(ticker, start_date, end_date)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok= True)
            data.to_csv(save_path)
            print(f"Saved mock stock data to {save_path}")

        return data

def collect_news_data(ticker, start_date, end_date, api_key, save_path = None):
    print(f"Collecting news data for {ticker} from {start_date} to {end_date}")
    
    try:
        import time
        time.sleep(2)
        
        company = yf.Ticker(ticker)
        
        try:
            company_name = company.info.get('shortName', ticker)
        except:
            print(f"Couldn't fetch company infor for {ticker}, using ticker as search item")
            company_name = ticker
    except Exception as e:
        print(f"Error getting company info: {str(e)}")
        company_name = ticker

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
            'language': 'en',
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
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
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

#Use for testing to bypass API limits, simulated testing
def generate_mock_news_data(ticker, start_date, end_date, save_path = None):
    print(f"Generating mock news data for {ticker} from {start_date} to {end_date}")
    
    import random
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    headlines = [
        f"{ticker} stock rises on strong earnings",
        f"{ticker} announces new product line",
        f"Analysts upgrade {ticker} rating",
        f"{ticker} faces regulatory challenges",
        f"Market uncertainty affects {ticker} performance",
        f"{ticker} reports quarterly results"
    ]
    
    mock_articles = []
    current_date = start_dt
    
    while current_date <= end_dt:
        if random.random() < .7:
            num_articles = random.randint(1, 3)
            
            for _ in range(num_articles):
                headline = random.choice(headlines)
                
                article = {
                    "title": headline,
                    "description": f"This is a mock description for {headline.lower()}.",
                    "content": f"This is mock content for an article about {ticker}. " * 3,
                    "publishedAt": current_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "source": {"name": random.choice(["Financial Times", "Bloomberg", "CNBC", "Reuters"])}
                }
                
                mock_articles.append(article)
            
        current_date += timedelta(days = 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok= True)
        with open(save_path, 'w') as f:
            json.dump(mock_articles, f)
        print(f"Saved mock news data to {save_path}")
    
    return mock_articles

def generate_mock_stock_data(ticker, start_date, end_date):
    print(f"Generating mock stock data for {ticker} from {start_date} to {end_date}")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_range = []
    current_date = start_dt
    
    while current_date <= end_dt:
        if current_date.weekday() < 5:
            date_range.append(current_date)
        current_date += timedelta(days = 1)
    
    start_price = 150.0
    volatility = .01
    
    prices = [start_price]
    
    for i in range(1, len(date_range)):
        daily_return = np.random.normal(.0005, volatility)
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    stock_data = pd.DataFrame(index=date_range)
    stock_data['Open'] = prices
    stock_data['High'] = stock_data['Open'] * (1 + np.random.uniform(0, 0.01, len(date_range)))
    stock_data['Low'] = stock_data['Open'] * (1 - np.random.uniform(0, 0.01, len(date_range)))
    stock_data['Close'] = stock_data['Open'] * (1 + np.random.normal(0, 0.005, len(date_range)))
    stock_data['Adj Close'] = stock_data['Close']
    stock_data['Volume'] = np.random.randint(5000000, 50000000, len(date_range))
    
    return stock_data