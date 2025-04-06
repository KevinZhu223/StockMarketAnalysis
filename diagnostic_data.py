# diagnostic_data.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def debug_data_structure(df, title="Data Structure Debug"):
    """Print detailed information about DataFrame structure"""
    print(f"\n=== {title} ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame index type: {type(df.index)}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Print column types
    print("\nColumn types:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Print first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check if columns are actually numeric
    print("\nSample values:")
    for col in df.columns:
        try:
            value = df[col].iloc[0]
            print(f"  - {col}: {value} (type: {type(value)})")
        except Exception as e:
            print(f"  - {col}: Error accessing value: {e}")

print("=== Stock Data Diagnostic ===")

# Try to load the problematic file
stock_path = 'data/processed/AAPL_stock_clean.csv'
if os.path.exists(stock_path):
    print(f"File exists: {stock_path}")
    
    # Load with different options to see what works
    print("\nLoading with default options:")
    df1 = pd.read_csv(stock_path)
    debug_data_structure(df1, "Default Loading")
    
    print("\nLoading with index_col=0:")
    df2 = pd.read_csv(stock_path, index_col=0)
    debug_data_structure(df2, "index_col=0")
    
    print("\nLoading with parse_dates=True:")
    df3 = pd.read_csv(stock_path, parse_dates=True)
    debug_data_structure(df3, "parse_dates=True")
    
    print("\nLoading with index_col=0, parse_dates=True:")
    df4 = pd.read_csv(stock_path, index_col=0, parse_dates=True)
    debug_data_structure(df4, "index_col=0, parse_dates=True")
    
    # Try to fix and save a corrected version
    print("\nAttempting to fix data:")
    try:
        df_fixed = df1.copy()
        
        # Check if we need to set the index
        if 'Date' in df_fixed.columns:
            df_fixed.set_index('Date', inplace=True)
        
        # Convert columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_fixed.columns:
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
        
        # Save the fixed version
        fixed_path = 'test_results/data/AAPL_fixed.csv'
        os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
        df_fixed.to_csv(fixed_path)
        print(f"Saved fixed data to {fixed_path}")
        
        debug_data_structure(df_fixed, "Fixed Data")
    except Exception as e:
        print(f"Error fixing data: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"File not found: {stock_path}")
    
    # Download fresh data as a reference
    print("\nDownloading fresh data for comparison:")
    try:
        import yfinance as yf
        df_yf = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
        debug_data_structure(df_yf, "yfinance Data")
        
        # Save reference data
        ref_path = 'test_results/data/AAPL_reference.csv'
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        df_yf.to_csv(ref_path)
        print(f"Saved reference data to {ref_path}")
    except Exception as e:
        print(f"Error downloading reference data: {e}")