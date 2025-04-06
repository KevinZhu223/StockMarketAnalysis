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
import random
import re

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
#Commenting this out
#news source not working since it doesn't allow for past news
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

def collect_finviz_news(ticker, save_path=None, max_retries=3, delay=2):
    print(f"Collecting Finviz news for {ticker}")

    # Using multiple user agents to avoid being blocked
    user_agents = [  # Changed to list instead of set
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    
    # Set up headers to mimic a real browser
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://finviz.com/",
        "Connection": "keep-alive"
    }
    
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    articles = []
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to fetch Finviz news")  # Fixed variable name
            
            time.sleep(delay + random.uniform(0, 1))
            
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 403:
                print("Access forbidden.")
                headers["User-Agent"] = random.choice(user_agents)
                continue
            if response.status_code != 200:
                print(f"Failed to retrieve Finviz page: Status code {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Save HTML for debugging
            debug_file = f"finviz_{ticker}_debug.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved HTML to {debug_file} for debugging")
            
            # Find news table - more flexible approach
            news_table = soup.find('table', class_='fullview-news-outer')
            
            if not news_table:
                print("No table with class 'fullview-news-outer' found. Trying alternative selectors...")
                
                # Try to find table by content pattern
                for table in soup.find_all('table'):
                    # Check if this looks like a news table
                    if table.find_all('a') and table.find_all('td'):
                        news_table = table
                        print("Found potential news table using alternative detection")
                        break
                
                if not news_table:
                    print("Could not find any table that looks like a news table")
                    continue
            
            rows = news_table.find_all('tr')
            print(f"Found {len(rows)} news rows")
            
            # Analyze first few rows to understand structure
            print("Analyzing row structure:")
            for i, row in enumerate(rows[:3]):  # Look at first 3 rows
                print(f"Row {i} structure:")
                for j, td in enumerate(row.find_all('td')):
                    print(f"  TD {j}: class='{td.get('class')}', align='{td.get('align')}', content: '{td.text.strip()[:30]}...'")
            
            # Process rows
            success_count = 0
            error_count = 0
            
            for row_idx, row in enumerate(rows):
                try:
                    # More flexible TD selection
                    all_tds = row.find_all('td')
                    
                    # Check if this row has at least 2 cells
                    if len(all_tds) >= 2:
                        # First cell usually contains the date
                        date_td = all_tds[0]
                        # Second cell usually contains the title
                        title_td = all_tds[1]
                        
                        date_text = date_td.text.strip()
                        title = title_td.text.strip()
                        
                        # Print what we found
                        print(f"Row {row_idx}: Found date '{date_text}' and title '{title[:30]}...'")
                        
                        # Extract link and source
                        link = title_td.a['href'] if title_td.a and 'href' in title_td.a.attrs else ""
                        source = title_td.span.text.strip() if title_td.span else "Unknown"
                        
                        pub_date = parse_finviz_date(date_text)
                        
                        article = {
                            "title": title,
                            "description": title,  # Finviz doesn't provide descriptions
                            "content": title,      # Use title as content
                            "publishedAt": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "source": {"name": source},
                            "url": link,
                            "ticker": ticker
                        }
                        
                        articles.append(article)
                        success_count += 1
                    else:
                        print(f"Row {row_idx}: Insufficient cells (found {len(all_tds)}, need at least 2)")
                        
                except Exception as e:
                    print(f"Error parsing row {row_idx}: {str(e)}")
                    error_count += 1
            
            print(f"Processed {len(rows)} rows: {success_count} successful, {error_count} errors")
            
            if articles:
                print(f"Successfully collected {len(articles)} articles from Finviz")
                break
            else:
                print("No articles were successfully extracted despite finding rows")
                
        except Exception as e:
            print(f"Error scraping Finviz news (attempt {attempt+1}): {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for better debugging
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    # Save collected articles
    if save_path and articles:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(articles, f)
        print(f"Saved {len(articles)} Finviz news items to {save_path}")
    else:
        print(f"No articles to save or no save path provided")

    return articles

def parse_finviz_date(date_text):
    """Parse Finviz date formats correctly."""
    today = datetime.now()
    
    if "Today" in date_text:
        # Extract time if available (format like "Today 03:58PM")
        time_part = date_text.replace("Today", "").strip()
        return datetime.combine(today.date(), parse_time(time_part))
    
    elif "Yesterday" in date_text:
        # Extract time if available
        time_part = date_text.replace("Yesterday", "").strip()
        yesterday = today - timedelta(days=1)
        return datetime.combine(yesterday.date(), parse_time(time_part))
    
    elif re.match(r"^\d{2}:\d{2}(AM|PM)$", date_text):
        # Just time (like "03:15PM") - assume it's today
        return datetime.combine(today.date(), parse_time(date_text))
    
    elif re.match(r"^[A-Za-z]{3}-\d{1,2}-\d{2}(\s\d{2}:\d{2}(AM|PM))?$", date_text):
        # Format like "Mar-30-25" or "Mar-30-25 09:59PM"
        date_parts = date_text.split()
        date_only = date_parts[0]
        
        month, day, year = date_only.split('-')
        year = int("20" + year)
        month_num = datetime.strptime(month, "%b").month
        day = int(day)
        
        date_obj = datetime(year, month_num, day)
        
        # If time part exists
        if len(date_parts) > 1:
            time_obj = parse_time(date_parts[1])
            return datetime.combine(date_obj.date(), time_obj)
        else:
            return date_obj
    
    # Default to current date if can't parse
    return today

def parse_time(time_text):
    """Parse time string like '03:58PM' to time object."""
    if not time_text:
        return datetime.now().time()
    
    try:
        # Remove any non-time characters
        time_text = re.sub(r'[^0-9:APM]', '', time_text)
        return datetime.strptime(time_text, "%I:%M%p").time()
    except ValueError:
        return datetime.now().time()
    
def collect_recent_data(ticker, days_back = 5):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"Collecting recent data for {ticker} from {start_date} to {end_date}")
    
    stock_path = f"data/realtime/{ticker}_stock_recent.csv"
    stock_data = collect_stock_data(ticker, start_date, end_date, save_path = stock_path)
    
    news_path = f"data/realtime/{ticker}_news_recent.json"
    news_data = collect_finviz_news(ticker, save_path=news_path)
    
    return stock_data, news_data, stock_path, news_path

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