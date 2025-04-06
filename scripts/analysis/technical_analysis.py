import pandas as pd
import numpy as np
from scripts.analysis.feature_engineering import add_tech_indicators

def calculate_advanced_indicators(df):
    df = add_tech_indicators(df)
    
    #Ichimoku Cloud
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2 #Conversion Line
    
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_Sen'] = (high_26 + low_26) / 2 #Base Line
    
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)  # Leading Span A
    
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)  # Leading Span B
    
    #Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = pd.Series(0, index = df.index)
    negative_flow = pd.Series(0, index=df.index)
    
    positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
    negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]

    #Calc MFI
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    mf_ratio = positive_mf / negative_mf
    df['MFI'] = 100 - (100 / (1 + mf_ratio))
    
    #Chaikin Oscillator
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    ad = clv * df['Volume']
    ad_line = ad.cumsum()
    
    df['Chaikin_Osc'] = ad_line.ewm(span=3).mean() - ad_line.ewm(span=10).mean()
    
    return df

def detect_candlestick_patterns(df):
    """Detect common candlestick patterns with error handling"""
    # Make a copy to avoid modifying the original
    df = df.copy()
       
    # Ensure all required columns are numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted {col} to numeric type")
    
    # Check if we have all required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        # Create synthetic data for missing columns
        for col in missing_cols:
            if col == 'Open':
                df[col] = df['Close'] * 0.99 if 'Close' in df.columns else 100
            elif col == 'High':
                df[col] = df['Close'] * 1.01 if 'Close' in df.columns else 101
            elif col == 'Low':
                df[col] = df['Close'] * 0.99 if 'Close' in df.columns else 99
            elif col == 'Close':
                df[col] = 100
       
    # Create copies of relevant columns with error handling
    try:
        df['BodySize'] = abs(df['Close'] - df['Open'])
        df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
           
        # Calculate average body size for reference
        avg_body = df['BodySize'].rolling(window=20).mean()
    
        df['Doji'] = df['BodySize'] < (.1 * avg_body)

        df['Hammer'] = (
            (df['Close'] > df['Open']) &
            (df['LowerShadow'] > 2 * df['BodySize']) &  # Long lower shadow
            (df['UpperShadow'] < 0.2 * df['BodySize'])  # Short upper shadow
        )
        
        #Shooting Star bearish
        df['ShootingStar'] = (
            (df['Close'] < df['Open']) &  # Bearish
            (df['UpperShadow'] > 2 * df['BodySize']) &  # Long upper shadow
            (df['LowerShadow'] < 0.2 * df['BodySize'])  # Short lower shadow
        )
        
        # Engulfing patterns
        df['BullishEngulfing'] = (
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous candle bearish
            (df['Close'] > df['Open']) &  # Current candle bullish
            (df['Open'] < df['Close'].shift(1)) &  # Open below previous close
            (df['Close'] > df['Open'].shift(1))  # Close above previous open
        )
        
        df['BearishEngulfing'] = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous candle bullish
            (df['Close'] < df['Open']) &  # Current candle bearish
            (df['Open'] > df['Close'].shift(1)) &  # Open above previous close
            (df['Close'] < df['Open'].shift(1))  # Close below previous open
        )
        
    except Exception as e:
           print(f"Error in candlestick pattern detection: {e}")
           # Return the dataframe with minimal patterns
           df['Doji'] = False
           df['Hammer'] = False
           df['ShootingStar'] = False
           df['BullishEngulfing'] = False
           df['BearishEngulfing'] = False
            
    return df