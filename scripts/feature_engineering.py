import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def add_tech_indicators(df):
    df_tech = df.copy()
    
    if 'Adj Close' in df_tech.columns and pd.api.types.is_numeric_dtype(df_tech['Adj Close']):
        price_col = 'Adj Close'
    elif 'Close' in df_tech.columns and pd.api.types.is_numeric_dtype(df_tech['Close']):
        price_col = 'Close'
    else:
        numeric_cols = [col for col in df_tech.columns
                        if pd.api.types.is_numeric_dtype(df_tech[col])]
        
        if numeric_cols:
            price_col = numeric_cols[0]
            print(f"Using {price_col} for price calcs")
        else:
            raise ValueError("No suitable price column")
    
    print(f"Calculating tech indicators using {price_col} as price column")
    
    #Moving Averages
    windows = [5,10,20,50]
    for window in windows:
        df_tech[f'SMA_{window}'] = df_tech[price_col].rolling(window = window).mean()
        
        #SMA Ratio (price relative to SMA)
        df_tech[f'SMA_{window}_Ratio'] = df_tech[price_col] / df_tech[f'SMA_{window}']
        
    #Exponential Moving Averages
    for window in [12,26]:
        df_tech[f'EMA_{window}'] = df_tech[price_col].ewm(span = window, adjust = False).mean()
        
    #MACD (Moving Average Convergence Divergence)
    df_tech['MACD'] = df_tech['EMA_12'] - df_tech['EMA_26']
    df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span = 9, adjust = False).mean()
    df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
    
    #Relative Stregth Index (RSI)
    delta = df_tech[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window = 14).mean()
    avg_loss = loss.rolling(window = 14).mean()
    
    rs = avg_gain / avg_loss
    
    df_tech['RSI_14'] = 100 - (100 / (1 + rs))
    
    #Bollinger bands
    df_tech['BB_Middle'] = df_tech[price_col].rolling(window = 20).mean()
    df_tech['BB_StdDev'] = df_tech[price_col].rolling(window = 20).std()
    df_tech['BB_Upper'] = df_tech['BB_Middle'] + (df_tech['BB_StdDev'] * 2)
    df_tech['BB_Lower'] = df_tech['BB_Middle'] - (df_tech['BB_StdDev'] *2)
    
    #BB band width (volatility)
    df_tech['BB_Width'] = (df_tech['BB_Upper'] - df_tech['BB_Lower']) / df_tech['BB_Middle']
    
    #Momentum Indicators
    for period in [5, 10, 21]:
        df_tech[f'Momentum_{period}'] = df_tech[price_col].pct_change(periods =period)
    
    #Volatility (rolling Standard Deviation)
    for window in [5, 21]:
        df_tech[f'Volatility_{window}'] = df_tech[price_col].pct_change().rolling(window = window).std()
        
    #Volume features
    if 'Volume' in df_tech.columns:
        #Volume moving average
        df_tech['Volume_SMA_5'] = df_tech['Volume'].rolling(window=5).mean()
        
        #Volume ratio
        df_tech['Volume_Ratio'] = df_tech['Volume'] / df_tech['Volume_SMA_5']
        
        #Price-volume relationship
        df_tech['Price_Volume_Ratio'] = df_tech[price_col] / df_tech['Volume']
    
    df_tech = df_tech.replace([np.inf, -np.inf], np.nan)
    
    return df_tech

def enhance_sentiment_features(df):
    df_sent = df.copy()
    
    if 'sentiment' not in df_sent.columns:
        raise ValueError("Sentiment column not found")
    
    #Chnage in sentiment
    df_sent['sentiment_change_1d'] = df_sent['sentiment'].diff(1)
    df_sent['sentiment_change_3d'] = df_sent['sentiment'].diff(3)
    
    
    #sentiment moving averages
    df_sent['sentiment_SMA_3'] = df_sent['sentiment'].rolling(window = 3).mean()
    df_sent['sentiment_SMA_7'] = df_sent['sentiment'].rolling(window = 7).mean()
    
    #Sentiment Volatility
    df_sent['sentiment_volatility'] = df_sent['sentiment'].rolling(window = 5).std()
    
    #Categorical sentimtent (pos, neg, neutral)
    df_sent['sentiment_category'] = pd.cut(
        df_sent['sentiment'],
        bins = [-1, -.2, .2,1],
        labels = ['negative', 'neutral', 'positive']
    )
    
    #One-hot encode sentiment categories
    sentiment_dummies = pd.get_dummies(df_sent['sentiment_category'], prefix = 'sentiment')
    df_sent = pd.concat([df_sent, sentiment_dummies], axis = 1)
    
    if 'article_count' in df_sent.columns:
        #normalize
        df_sent['article_count_normalized'] = df_sent['article_count'] / df_sent['article_count'].mean()
        
        #article count momentum
        df_sent['article_count_change'] = df_sent['article_count'].diff(1)
        
        #moving average
        df_sent['article_count_SMA_3'] = df_sent['article_count'].rolling(window = 3).mean()
    
    return df_sent       

def combine_stock_sentiment(stock_df, sentiment_df):
    stock = stock_df.copy()
    sentiment = sentiment_df.copy()
    
    # Print the index values to debug
    print("Stock index sample:", stock.index[:3])
    print("Sentiment index sample:", sentiment.index[:3])
    
    # Check if 'Date' is in the index (common issue)
    if 'Date' in stock.index:
        print("Found 'Date' in stock index, removing it...")
        stock = stock[stock.index != 'Date']
    
    if 'Date' in sentiment.index:
        print("Found 'Date' in sentiment index, removing it...")
        sentiment = sentiment[sentiment.index != 'Date']
    
    #Covert to datetime    
    if not isinstance(stock.index, pd.DatetimeIndex):
        try:
            stock.index = pd.to_datetime(stock.index, errors='coerce')
            # Drop rows where date conversion failed
            stock = stock[~stock.index.isna()]
        except Exception as e:
            print(f"Error converting stock index to datetime: {e}")
            print("Stock index values:", stock.index.tolist()[:5])
            return None
        
    if not isinstance(sentiment.index, pd.DatetimeIndex):
        try:
            sentiment.index = pd.to_datetime(sentiment.index, errors='coerce')
            # Drop rows where date conversion failed
            sentiment = sentiment[~sentiment.index.isna()]
        except Exception as e:
            print(f"Error converting sentiment index to datetime: {e}")
            print("Sentiment index values:", sentiment.index.tolist()[:5])
            return None
    
    combined_df = stock.join(sentiment, how = 'left')
    
    sentiment_cols = [col for col in sentiment.columns]
    combined_df[sentiment_cols] = combined_df[sentiment_cols].fillna(method = 'ffill')
    
    if 'Adj Close' in combined_df.columns:
        price_col = 'Adj Close'
    elif 'Close' in combined_df.columns:
        price_col = 'Close'
    else:
        numeric_cols = [col for col in combined_df.columns
                        if pd.api.types.is_numeric_dtype(combined_df[col])]
        
        if numeric_cols:
            price_col = numeric_cols[0]
        else:
            raise ValueError("No suitable price column")
        
    if 'sentiment' in combined_df.columns:
        combined_df['price_sentiment'] = combined_df[price_col] * combined_df['sentiment']
        
        if 'Daily_Return' in combined_df.columns:
            combined_df['return_sentiment'] = combined_df['Daily_Return'] * combined_df['sentiment']
            
        if 'Volatility_21' in combined_df.columns:
            combined_df['volatility_sentiment'] = combined_df['Volatility_21'] * combined_df['sentiment']
            
        if 'RSI_14' in combined_df.columns:
            combined_df['rsi_sentiment'] = combined_df['RSI_14'] * combined_df['sentiment']
            
        if 'MACD' in combined_df.columns:
            combined_df['macd_sentiment'] = combined_df['MACD'] * combined_df['sentiment']
            
        #Creating lagged features for prediction
        #Use to predict future price movements
        
        combined_df['target_next_day_return'] = combined_df['Daily_Return'].shift(-1)
        combined_df['target_next_day_direction'] = (combined_df['target_next_day_return'] > 0).astype(int)
        
        #multi-day predictions
        combined_df['target_3day_return'] = combined_df[price_col].pct_change(periods = 3).shift(-3)
        combined_df['target_5day_return'] = combined_df[price_col].pct_change(periods = 5).shift(-5)
        
        #binary targets for multi-day direction
        combined_df['target_3day_direction'] = (combined_df['target_3day_return'] > 0).astype(int)
        combined_df['target_5day_direction'] = (combined_df['target_5day_return'] > 0).astype(int)
        
    return combined_df
    
def process_features(ticker, stock_path, sentiment_path, output_path = None):
    print(f"Processing features for {ticker}")
    
    #load data
    try:
        stock_df = pd.read_csv(stock_path, index_col = 0, parse_dates = True)
        print(f"Loaded stock data with shape: {stock_df.shape}")
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None
    
    try:
        sentiment_df = pd.read_csv(sentiment_path, index_col = 0, parse_dates= True)
        print(f"Loaded sentiment data with shape: {sentiment_df.shape}")
    except Exception as e:
        print(f"Error loading sentiment data: {str(e)}")
        return None
    
    # add tech indicators
    try:
        stock_with_indicators = add_tech_indicators(stock_df)
        print(f"Added tech indicators, new shape: {stock_with_indicators.shape}")
    except Exception as e:
        print(f"Error adding tech indicators: {str(e)}")
        return None
    
    #enhance sentiment features
    try:
        enhanced_sentiment = enhance_sentiment_features(sentiment_df)
        print(f"Enhanced sentiment features, new shape: {enhanced_sentiment.shape}")
    except Exception as e:
        print(f"Error enhancing sentiment features: {str(e)}")
        return None
    
    #Combine datasets
    try:
        combined_df = combine_stock_sentiment(stock_with_indicators, enhanced_sentiment)
        print(f"Combined data, final shape: {combined_df.shape}")
    except Exception as e:
        print(f"Error combining datasets: {str(e)}")
        return None

    #Drop rows with NaN values in key columns
    target_cols = ['target_next_day_direction', 'target_3day_direction', 'target_5day_direction']
    combined_df = combined_df.dropna(subset=target_cols)
    print(f"Shape after dropping NaN targets: {combined_df.shape}")
    
    #save data
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path)
        print(f"Saved processed features to {output_path}")
    
    return combined_df

def process_all_tickers(tickers, base_dir = 'data'):
    for ticker in tickers:
        stock_path = f"{base_dir}/processed/{ticker}_stock_clean.csv"
        sentiment_path = f"{base_dir}/processed/{ticker}_news_clean.csv"
        output_path = f"{base_dir}/final/{ticker}_features.csv"
        
        process_features(ticker, stock_path, sentiment_path, output_path)
        print(f"Completed feature processing for {ticker}")
    
# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    process_all_tickers(tickers)