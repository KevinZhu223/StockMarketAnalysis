import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

#Download nltk resouces only once
#nltk.download('punkt')
#nltk.download('stopwords')

def clean_stock_data(file_path, output_path = None, ticker = None):
    print(f"Cleaning stock data from {file_path}")
    
    if ticker is None:
        import re
        match = re.search(r'([A-Z]+)_stock', file_path)
        if match:
            ticker = match.group(1)
            print(f"Extracted ticker '{ticker}' from file path")
            
    try:
        with open(file_path, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        print(f"First few lines of the file:\n{''.join(first_lines)}")
        
        has_ticker_row = any("Ticker" in line for line in first_lines)
        has_header_mismatch = "Price" in first_lines[0] and "Date" in ''.join(first_lines[:3])
        
        if has_ticker_row and has_header_mismatch:
            print("Detected non-standard CSV format")
            df = pd.read_csv(file_path, skiprows = 1, index_col=0, parse_dates = True)
        else:
            try:
                df = pd.read_csv(file_path, index_col = 0, parse_dates = True)
            
                if "Ticker" in df.index:
                    print("Found Ticker")
                    df = pd.read_csv(file_path, skiprows = 1, index_col = 0, parse_dates = True)
            except Exception as e:
                print(f"Standard loading failed")
                df = pd.read_csv(file_path, skiprows = [1,2], index_col = 0, parse_dates= True)
        
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types: {df.dtypes}")
        
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        if ticker and all(col.startswith(ticker) or col.startswith(f"{ticker}.") for col in df.columns[:6]):
            print(f"Detected ticker-based column names. Starardizing...")
            financial_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volumn']
            
            #Mapping dict
            rename_dict = {}
            for i, std_col in enumerate(financial_cols):
                if i < len(df.columns):
                    rename_dict[df.columns[i]] = std_col
                
            #Rename columns
            df = df.rename(columns =rename_dict)
            print(f"Renamed columns to: {df.columns.tolist()}")
        elif not any(col in df.columns for col in standard_cols):
            print("No standard column names found.")
            financial_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            
            num_cols = min(len(df.columns), len(financial_cols))
            rename_dict = {df.columns[i]: financial_cols[i] for i in range(num_cols)}
            df = df.rename(columns = rename_dict)
            print(f"Inferred column names: {df.columns.tolist()}")
            
        # Convert all numeric columns to float
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {str(e)}")      
                
        print(f"Data types after conversion: {df.dtypes}")
        
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values")
            
            #Forward fill missing values with previous days data
            df.fillna(method = 'ffill', inplace = True)

            #Fill any missing values at the beginning using backward fill
            df.fillna(method = 'bfill', inplace = True)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                invalid_dates = df.index.isna()
                if any(invalid_dates):
                    print(f"Dropping {sum(invalid_dates)} rows with invalid dates")
                    df = df[~invalid_dates]
            except Exception as e:
                print(f"Error coverting index to datetime: {str(e)}")
                
        if 'Adj Close' in df.columns and pd.api.types.is_numeric_dtype(df['Adj Close']):
            df['Daily_Return'] = df['Adj Close'].pct_change()
            price_col_used = 'Adj Close'
        elif 'Close' in df.columns and pd.api.types.is_numeric_dtype(df['Close']):
            print("Using 'Close' insted of 'Adj Close'")
            df['Daily_Return'] = df['Close'].pct_change()
            price_col_used = 'Close'
        else:
            print("Neither columns available")
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                price_col_used = numeric_cols[0]
                print(f"Using {price_col_used} for returns calculation")
                df['Daily_Return'] = df[price_col_used].pct_change()
            else:
                print("No numeric columns available") 
                df['Daily_Return'] = 0  
                price_col_used = None

        #Handling outliers in daily returns using IQR method to find extreme price movements
        if 'Daily_Return' in df.columns and not df['Daily_Return'].isna().all():  
            Q1 = df['Daily_Return'].quantile(.25)
            Q3 = df['Daily_Return'].quantile(.75)
            IQR = Q3 - Q1
                
            #Flag the outliers
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df['Return_Outlier'] = ((df['Daily_Return'] < lower_bound) |
                                    (df['Daily_Return'] > upper_bound))
        else:
            df['Return_Outlier'] = False
            
        #Data Validation
        required_cols = ['Daily_Return', 'Return_Outlier']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        
        if len(df) == 0:
            print("Warning: Dataframe empty")
        elif price_col_used and df[price_col_used].isna().all():
            print(f"Warning: All values in {price_col_used} are NaN")
        
        
        #Save clean data to ouput path
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok = True)
            df.to_csv(output_path)
            print(f"Saved cleaned stock data to {output_path}")
            
            # Also save a summary of the data cleaning process
            summary_path = output_path.replace('.csv', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Data Cleaning Summary for {ticker}\n")
                f.write(f"Original columns: {df.columns.tolist()}\n")
                f.write(f"Standardized columns: {df.columns.tolist()}\n")
                f.write(f"Price column used: {price_col_used}\n")
                f.write(f"Data shape: {df.shape}\n")
                f.write(f"Date range: {df.index.min()} to {df.index.max()}\n")
                f.write(f"Missing values before cleaning: {missing_values}\n")
                f.write(f"Outliers detected: {df['Return_Outlier'].sum()}\n")
            print(f"Saved cleaning summary to {summary_path}")  
              
        return df

    except Exception as e:
        print(f"Error in clean_stock_data: {str(e)}")
        
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'Return_Outlier'])
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path)
            print(f"Saved empty dataframe to {output_path}")
        
        return df
            

def clean_text(text):
    
    if not isinstance(text, str) or text is None:
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    text = re.sub(r'</*?>', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def analyze_sentiment(text):
    
    if not text:
        return 0.0
    
    try:
        analysis = TextBlob(text)
        
        return analysis.sentiment.polarity
    except ImportError:
        positive_words = ['up', 'rise', 'gain', 'profit', 'growth', 'positive', 
                        'increase', 'higher', 'bull', 'bullish', 'opportunity']
        
        negative_words = ['down', 'fall', 'loss', 'decline', 'negative', 'decrease',
                        'lower', 'bear', 'bearish', 'risk', 'concern']
        
        pos_count = sum(1 for word in text.split() if word in positive_words)
        neg_count = sum(1 for word in text.split() if word in negative_words)
        
        total_count = pos_count + neg_count
        
        if total_count == 0:
            return 0
        
        return (pos_count - neg_count) / total_count

def clean_news_data(file_path, output_path = None):
    print(f"Cleaning news data from {file_path}")
    
    with open(file_path, 'r') as f:
        news_data = json.load(f)
        
    print(f"Loaded {len(news_data)} news articles")
    
    processed_articles = []
    
    for article in news_data:
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        
        full_text = f"{title} {description} {content}"
        
        cleaned_text = clean_text(full_text)
        
        if len(cleaned_text) < 10:
            continue
        
        try:
            date_str = article.get('publishedAt', '')
            if 'T' in date_str:
                # Format: 2023-01-01T12:00:00Z
                date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d').strftime('%Y-%m-%d')
            else:
                date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        
        sentiment_score = analyze_sentiment(cleaned_text)

        processed_article = {
            'date': date,
            'title': title,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment_score,
            'source': article.get('source', {}).get('name', 'Unknown')
        }
        
        processed_articles.append(processed_article)
    
    print(f"Successfully processed {len(processed_articles)} articles")
    
    df = pd.DataFrame(processed_articles)
    
    df['date'] = pd.to_datetime(df['date'])
    
    daily_sentiment = df.groupby(df['date'].dt.date).agg({
        'sentiment': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'article_count'})
    
    #Convert index to datetime
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok= True)
        daily_sentiment.to_csv(output_path)
        print(f"Saved cleaned news data to {output_path}")
        
    return daily_sentiment

def clean_data_for_tickers(tickers):
    for ticker in tickers:
        stock_input = f"data/raw_data/{ticker}_stock.csv"
        stock_output = f"data/clean_data/{ticker}_stock_clean.csv"
        
        if os.path.exists(stock_input):
            clean_stock_data(stock_input, stock_output)
        else:
            print(f"Warning: {stock_input} not found")
        
        news_input = f"data/raw_data/{ticker}_news.json"
        news_output = f"data/clean_data/{ticker}_news_clean.csv"
        
        if os.path.exists(news_input):
            clean_news_data(news_input, news_output)
        else:
            print(f"Warning: {news_input} not found")
        
        print(f"Completed data cleaning for {ticker}")
    

# Example usage
if __name__ == "__main__":
    # List of tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    clean_data_for_tickers(tickers)