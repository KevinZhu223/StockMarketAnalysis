import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Download nltk resouces only once
nltk.download('punkt')
nltk.download('stopwords')

def clean_stock_data(file_path, output_path = None):
    print(f"Cleaning stock data from {file_path}")
    
    df = pd.read_csv(file_path, index_col = 0, parse_dates = True)
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values")
        
        #Forward fill missing values with previous days data
        df.fillna(method = 'ffill', inplace = True)

        #Fill any missing values at the beginning using backward fill
        df.fillna(method = 'bfill', inplace = True)
        
    #Daily returns
    df['Daily_Return'] = df['Adj Close'].pct_change()
        
    #Handling outliers in daily returns using IQR method to find extreme price movements
    Q1 = df['Daily_Return'].quantile(.25)
    Q3 = df['Daily_Return'].quantile(.75)
    IQR = Q3 - Q1
        
    #Flag the outliers
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df['Return_Outlier'] = ((df['Daily_Return'] < lower_bound) |
                            (df['Daily_Return'] > upper_bound))
        
    #Save clean data to ouput path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        df.to_csv(output_path)
        print(f"Saved cleaned stock data to {output_path}")
            
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
                date = datetime.strptime(date.str, '%Y-%m-%d').strftime('%Y-%m-%d')
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