# test_feature_engineering.py
import os
import sys
import pandas as pd
from scripts.feature_engineering import process_features
import matplotlib.pyplot as plt
import seaborn as sns

# Test with a single ticker
test_ticker = 'AAPL'

# Define paths
stock_path = f"data/test/processed/{test_ticker}_stock_clean.csv"
sentiment_path = f"data/test/processed/{test_ticker}_news_clean.csv"
output_path = f"data/test/final/{test_ticker}_features_realtime.csv"

# Create output directory
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Process features
print(f"Testing real-time feature engineering for {test_ticker}...")
features_df = process_features(test_ticker, stock_path, sentiment_path, output_path)

# Verify results
if features_df is not None:
    print("\nFeature Engineering Results:")
    print(f"Total features created: {features_df.shape[1]}")
    
    # Display feature categories
    tech_indicators = [col for col in features_df.columns if any(x in col for x in ['SMA', 'EMA', 'MACD', 'RSI', 'BB_', 'Momentum', 'Volatility'])]
    sentiment_features = [col for col in features_df.columns if 'sentiment' in col]
    interaction_features = [col for col in features_df.columns if any(x in col for x in ['_sentiment', 'price_sentiment', 'return_sentiment'])]
    target_features = [col for col in features_df.columns if 'target' in col]
    
    print(f"Technical indicators: {len(tech_indicators)}")
    print(f"Sentiment features: {len(sentiment_features)}")
    print(f"Interaction features: {len(interaction_features)}")
    print(f"Target features: {len(target_features)}")
    
    # Display sample data
    print("\nSample of processed features:")
    if not features_df.empty:
        print(features_df.head())
    else:
        print("Features DataFrame emtpy")
            
    # Check for missing values
    missing_values = features_df.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    if not columns_with_missing.empty:
        print(f"\nColumns with missing values")
        print(columns_with_missing)
    else:
        print("\nNo missing values found in any columns")
    
    #Directory for visualization
    os.makedirs("results", exist_ok = True)
    
    #Correlation heatmap
    plt.figure(figsize=(12,10))
    
    important_features = []
    important_features.extend(tech_indicators[:5] if len(tech_indicators) > 5 else tech_indicators)
    important_features.extend(sentiment_features[:3] if len(sentiment_features) > 3 else sentiment_features)
    important_features.extend(target_features)
    
    #Calculate and plot correlation matrix
    if important_features and len(features_df) > 5:
        corr_matrix = features_df[important_features].corr()
        sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", linewidth = .5)
        plt.title(f"{test_ticker} - Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"results/{test_ticker}_correlation.png")
        print(f"Saved correlation heatmap to results/{test_ticker}_correlation.png")

    #Plot price with sentiment
    plt.figure(figsize=(12,6))

    # Find a suitable price column
    price_col = None
    if 'Adj Close' in features_df.columns:
        price_col = 'Adj Close'
    elif 'Close' in features_df.columns:
        price_col = 'Close'
    else:
        # Look for ticker-specific columns that might contain price data
        numeric_cols = [col for col in features_df.columns 
                    if pd.api.types.is_numeric_dtype(features_df[col])]
        if numeric_cols:
            # Use the first numeric column as price (usually the one used in feature engineering)
            price_col = numeric_cols[0]
            print(f"Using {price_col} as price column for plotting")

    if price_col:
        ax1 = plt.gca()
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.plot(features_df.index, features_df[price_col], color='blue', label=f'Price ({price_col})')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        if 'sentiment' in features_df.columns:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Sentiment', color='red')
            ax2.plot(features_df.index, features_df['sentiment'], color='red', label='Sentiment')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add a legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f"{test_ticker} - Price and Sentiment")
        plt.tight_layout()
        plt.savefig(f"results/{test_ticker}_price_sentiment.png")
        print(f"Saved price and sentiment plot to results/{test_ticker}_price_sentiment.png")
    else:
        print("No suitable price column found for plotting")


    # Count of each target class
    if 'target_next_day_direction' in features_df.columns:
        up_days = features_df['target_next_day_direction'].sum()
        total_days = len(features_df['target_next_day_direction'])
        print(f"\nTarget distribution (next day): {up_days} up days ({up_days/total_days:.1%}), {total_days-up_days} down days ({(total_days-up_days)/total_days:.1%})")
else:
    print("Feature engineering failed!")