# test_fixes.py
import pandas as pd
import numpy as np
import os

# Create test data
dates = pd.date_range(start='2022-01-01', periods=100)
test_data = pd.DataFrame({
    'Open': np.random.randn(100).cumsum() + 100,
    'High': np.random.randn(100).cumsum() + 102,
    'Low': np.random.randn(100).cumsum() + 98,
    'Close': np.random.randn(100).cumsum() + 100,
    'Volume': np.random.randint(1000, 10000, 100)
}, index=dates)

# Save test data
os.makedirs('test_data', exist_ok=True)
test_data.to_csv('test_data/synthetic_stock_data.csv')

# Import and test functions
from scripts.analysis.feature_engineering import add_tech_indicators
from scripts.analysis.technical_analysis import calculate_advanced_indicators, detect_candlestick_patterns

# Test add_tech_indicators
print("\nTesting add_tech_indicators...")
indicators_df = add_tech_indicators(test_data)
print(f"Success! New shape: {indicators_df.shape}")
print(f"New columns: {[col for col in indicators_df.columns if col not in test_data.columns][:5]}...")

# Test calculate_advanced_indicators
print("\nTesting calculate_advanced_indicators...")
advanced_df = calculate_advanced_indicators(test_data)
print(f"Success! New shape: {advanced_df.shape}")
print(f"Advanced columns: {[col for col in advanced_df.columns if col not in indicators_df.columns][:5]}...")

# Test detect_candlestick_patterns
print("\nTesting detect_candlestick_patterns...")
patterns_df = detect_candlestick_patterns(test_data)
print(f"Success! New shape: {patterns_df.shape}")
print(f"Pattern columns: {[col for col in patterns_df.columns if col in ['Doji', 'Hammer', 'ShootingStar', 'BullishEngulfing', 'BearishEngulfing']]}")

print("\nAll tests completed successfully!")