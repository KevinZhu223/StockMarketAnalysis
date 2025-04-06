# realtime_analysis.py
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import your modules
from scripts.data.data_collectionction import collect_recent_data
from scripts.data.data_cleaning import clean_stock_data, clean_news_data
from scripts.analysis.feature_engineering import process_features

def run_realtime_analysis(ticker, days_back=5):
    """
    Run a complete real-time analysis pipeline for the given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        Number of days to analyze
    """
    print(f"Running real-time analysis for {ticker} over the past {days_back} days")
    
    # Create directories
    os.makedirs("data/realtime", exist_ok=True)
    os.makedirs("results/realtime", exist_ok=True)
    
    # Step 1: Collect recent data
    _, _, stock_path, news_path = collect_recent_data(ticker, days_back)
    
    # Step 2: Clean data
    stock_clean_path = f"data/realtime/{ticker}_stock_clean.csv"
    news_clean_path = f"data/realtime/{ticker}_news_clean.csv"
    
    clean_stock_data(stock_path, stock_clean_path, ticker)
    clean_news_data(news_path, news_clean_path, days_back)
    
    # Step 3: Feature engineering
    features_path = f"data/realtime/{ticker}_features.csv"
    features_df = process_features(ticker, stock_clean_path, news_clean_path, features_path)
    
    if features_df is not None:
        # Step 4: Generate visualizations
        generate_realtime_report(ticker, features_df, days_back)
        return features_df
    else:
        print("Error: Feature engineering failed.")
        return None

def generate_realtime_report(ticker, features_df, days_back):
    """
    Generate visualizations and reports for real-time analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    features_df : pandas.DataFrame
        Processed features dataframe
    days_back : int
        Number of days analyzed
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create output directory
    output_dir = "results/realtime"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price and Sentiment Chart
    plt.figure(figsize=(12, 6))
    
    # Find appropriate price column
    price_col = None
    for col in ['Adj Close', 'Close']:
        if col in features_df.columns:
            price_col = col
            break
    
    if price_col is None:
        price_col = features_df.select_dtypes(include=[np.number]).columns[0]
    
    # Plot price
    ax1 = plt.gca()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)', color='blue')
    ax1.plot(features_df.index, features_df[price_col], color='blue', label=f'Price ({price_col})')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot sentiment if available
    if 'sentiment' in features_df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sentiment Score', color='red')
        ax2.plot(features_df.index, features_df['sentiment'], color='red', label='Sentiment')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add a zero line for sentiment
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f"{ticker} - Recent Price and Sentiment (as of {timestamp})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}_recent_price_sentiment.png")
    plt.close()
    
    # 2. Correlation Matrix
    # Select important features
    tech_indicators = [col for col in features_df.columns 
                       if any(x in col for x in ['SMA', 'EMA', 'MACD', 'RSI', 'BB_', 'Momentum', 'Volatility'])]
    sentiment_features = [col for col in features_df.columns 
                         if 'sentiment' in col]
    target_features = [col for col in features_df.columns 
                      if 'target' in col]
    
    important_features = []
    
    # Add technical indicators
    if tech_indicators:
        tech_to_include = min(5, len(tech_indicators))
        important_features.extend(tech_indicators[:tech_to_include])
    
    # Add sentiment features
    if sentiment_features:
        sent_to_include = min(3, len(sentiment_features))
        important_features.extend(sentiment_features[:sent_to_include])
    
    # Add target features
    if target_features:
        important_features.extend(target_features)
    
    # Create correlation matrix if we have enough data
    if len(important_features) > 1 and len(features_df) > 3:
        plt.figure(figsize=(12, 10))
        corr_matrix = features_df[important_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidth=.5)
        plt.title(f"{ticker} - Feature Correlation Matrix (Real-time Analysis)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ticker}_correlation.png")
        plt.close()
    
    # 3. Summary Report
    with open(f"{output_dir}/{ticker}_summary.txt", 'w') as f:
        f.write(f"Real-time Analysis for {ticker}\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write(f"Period: Last {days_back} days\n\n")
        
        if len(features_df) > 0:
            f.write("Price Statistics:\n")
            f.write(f"Current Price: ${features_df[price_col].iloc[-1]:.2f}\n")
            
            if len(features_df) > 1:
                price_change = (features_df[price_col].iloc[-1] / features_df[price_col].iloc[0] - 1) * 100
                f.write(f"Price Change: {price_change:.2f}%\n")
            
            if 'sentiment' in features_df.columns:
                f.write("\nSentiment Statistics:\n")
                f.write(f"Current Sentiment: {features_df['sentiment'].iloc[-1]:.4f}\n")
                f.write(f"Average Sentiment: {features_df['sentiment'].mean():.4f}\n")
                
                if len(features_df) > 2 and 'sentiment' in features_df.columns:
                    correlation = features_df[[price_col, 'sentiment']].corr().iloc[0, 1]
                    f.write(f"Price-Sentiment Correlation: {correlation:.4f}\n")
    
    print(f"Real-time analysis report generated in {output_dir}/")

if __name__ == "__main__":
    # Default values
    ticker = "AAPL"
    days_back = 5
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    if len(sys.argv) > 2:
        days_back = int(sys.argv[2])
    
    # Run analysis
    run_realtime_analysis(ticker, days_back)