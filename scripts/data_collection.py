#Downloads a single stock
import os
import requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import json
from dotenv import load_dotenv
import numpy as np
from bs4 import BeautifulSoup
import requests
import time

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

def collect_finviz_news(ticker, save_path = None):
    print(f"Collecting Finviz news for {ticker}")

    headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    
    try:
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f"Failed to retrieve Finviz page: Status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        #find news table
        news_table = soup.find('table', class_= 'fullview-news-outer')
        
        if not news_table:
            print("No news table found")
            return []
        
        rows = news_table.find_all('tr')
        
        articles = []
        for row in rows:
            try:
                date_td = row.find('td', class_='fullview-news-td')
                title_td = row.find('td', class_='fullview-news-td', align='left')
                
                if date_td and title_td:
                    date_text = date_td.text.strip()
                    title = title_td.text.strip()
                    link = title_td.a['href']
                    source = title_td.span.text.strip()
                    
                    #Parse date
                    today = datetime.now().date()
                    if "Today" in date_text:
                        pub_date = datetime.now()
                    elif date_text.startswith("Yesterday"):
                        pub_date = datetime.now() - timedelta(days=1)
                    else:
                        try:
                            date_parts = date_text.split('-')
                            month = date_parts[0]
                            day = int(date_parts[1])
                            year = int("20" + date_parts[2])
                            pub_date = datetime(year, datetime.strptime(month, "%b").month, day)
                        except Exception as date_error:
                            print(f"Error parsing date '{date_text}': {str(date_error)}")
                            pub_date = datetime.now()
                    article = {
                         "title": title,
                           "description": title,
                           "content": title,  # Finviz doesn't provide content
                           "publishedAt": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "source": {"name": source},
                           "url": link,
                           "ticker": ticker
                    }
                    
                    articles.append(article)
            except Exception as e:
                print(f"Error parsing Finviz news item: {str(e)}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                   json.dump(articles, f)
            print(f"Saved {len(articles)} Finviz news items to {save_path}")

        return articles
    except Exception as e:
        print(f"Error scraping Finviz news: {str(e)}")
        return []
    
def collect_news_with_fallback(ticker, start_date, end_date, api_key=None, save_path=None):
       """Try to collect news from Finviz, fall back to mock data if it fails."""
       print(f"Collecting news for {ticker} with fallback")
       
       # Try Finviz first
       try:
           articles = collect_finviz_news(ticker, save_path=None)  # Don't save yet
           
           if articles and len(articles) >= 3:  # Consider successful if we get at least 3 articles
               print(f"Successfully collected {len(articles)} articles from Finviz")
               
               # Save if needed
               if save_path:
                   os.makedirs(os.path.dirname(save_path), exist_ok=True)
                   with open(save_path, 'w') as f:
                       json.dump(articles, f)
                   print(f"Saved Finviz news to {save_path}")
                   
               return articles
       except Exception as e:
           print(f"Finviz collection failed: {str(e)}")
       
       # If we get here, Finviz failed, use mock data
       print("Falling back to mock news data")
       return generate_mock_news_data(ticker, start_date, end_date, save_path)
   
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