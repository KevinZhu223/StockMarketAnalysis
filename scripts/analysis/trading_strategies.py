# scripts/analysis/trading_strategies.py

import pandas as pd
import numpy as np

def define_trading_strategies(features_df):
    """Define various trading strategies based on technical indicators"""
    strategies = {}
    
    # 1. Moving Average Crossover
    if all(col in features_df.columns for col in ['SMA_20', 'SMA_50']):
        strategies['MA_Crossover'] = {
            'signal': features_df['SMA_20'] > features_df['SMA_50'],
            'description': 'Buy when 20-day SMA crosses above 50-day SMA, sell when it crosses below'
        }
    
    # 2. RSI Strategy
    if 'RSI_14' in features_df.columns:
        strategies['RSI'] = {
            'signal': (features_df['RSI_14'] < 30) | ((features_df['RSI_14'].shift(1) < 30) & (features_df['RSI_14'] > 30)),
            'description': 'Buy when RSI goes below 30 (oversold) or crosses back above 30, sell when RSI goes above 70'
        }
    
    # 3. MACD Strategy
    if all(col in features_df.columns for col in ['MACD', 'MACD_Signal']):
        strategies['MACD'] = {
            'signal': features_df['MACD'] > features_df['MACD_Signal'],
            'description': 'Buy when MACD crosses above signal line, sell when it crosses below'
        }
    
    # 4. Bollinger Band Strategy
    if all(col in features_df.columns for col in ['BB_Lower', 'BB_Upper', 'Close']):
        strategies['Bollinger'] = {
            'signal': features_df['Close'] < features_df['BB_Lower'],
            'description': 'Buy when price touches lower Bollinger Band, sell when it touches upper band'
        }
    
    # 5. Combined Strategy (example: RSI + MACD)
    if 'RSI_14' in features_df.columns and all(col in features_df.columns for col in ['MACD', 'MACD_Signal']):
        strategies['RSI_MACD_Combined'] = {
            'signal': (features_df['RSI_14'] < 40) & (features_df['MACD'] > features_df['MACD_Signal']),
            'description': 'Buy when RSI is below 40 AND MACD is above signal line'
        }
    
    return strategies

def backtest_strategy(features_df, strategy_signal, initial_capital=10000):
    """Backtest a trading strategy"""
    # Create backtest dataframe
    backtest = pd.DataFrame(index=features_df.index)
    backtest['Close'] = features_df['Close']
    backtest['Signal'] = strategy_signal.astype(int)
    
    # Calculate returns
    backtest['Returns'] = backtest['Close'].pct_change()
    
    # Calculate strategy returns (with 1-day lag to avoid look-ahead bias)
    backtest['Strategy_Signal'] = backtest['Signal'].shift(1)
    backtest['Strategy_Returns'] = backtest['Returns'] * backtest['Strategy_Signal']
    
    # Calculate cumulative returns
    backtest['Cumulative_Returns'] = (1 + backtest['Returns']).cumprod()
    backtest['Strategy_Cumulative'] = (1 + backtest['Strategy_Returns']).cumprod()
    
    # Calculate equity curves
    backtest['Buy_Hold_Equity'] = initial_capital * backtest['Cumulative_Returns']
    backtest['Strategy_Equity'] = initial_capital * backtest['Strategy_Cumulative']
    
    # Calculate drawdowns
    backtest['Buy_Hold_Peak'] = backtest['Buy_Hold_Equity'].cummax()
    backtest['Strategy_Peak'] = backtest['Strategy_Equity'].cummax()
    
    backtest['Buy_Hold_Drawdown'] = (backtest['Buy_Hold_Equity'] / backtest['Buy_Hold_Peak']) - 1
    backtest['Strategy_Drawdown'] = (backtest['Strategy_Equity'] / backtest['Strategy_Peak']) - 1
    
    # Calculate performance metrics
    total_days = len(backtest)
    trading_days_per_year = 252
    years = total_days / trading_days_per_year
    
    # Strategy metrics
    strategy_return = backtest['Strategy_Equity'].iloc[-1] / initial_capital - 1
    strategy_annual_return = (1 + strategy_return) ** (1 / years) - 1
    strategy_volatility = backtest['Strategy_Returns'].std() * np.sqrt(trading_days_per_year)
    strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility > 0 else 0
    strategy_max_drawdown = backtest['Strategy_Drawdown'].min()
    
    # Buy & Hold metrics
    bh_return = backtest['Buy_Hold_Equity'].iloc[-1] / initial_capital - 1
    bh_annual_return = (1 + bh_return) ** (1 / years) - 1
    bh_volatility = backtest['Returns'].std() * np.sqrt(trading_days_per_year)
    bh_sharpe = bh_annual_return / bh_volatility if bh_volatility > 0 else 0
    bh_max_drawdown = backtest['Buy_Hold_Drawdown'].min()
    
    # Compile results
    results = {
        'Strategy': {
            'Total_Return': strategy_return,
            'Annual_Return': strategy_annual_return,
            'Volatility': strategy_volatility,
            'Sharpe_Ratio': strategy_sharpe,
            'Max_Drawdown': strategy_max_drawdown
        },
        'Buy_Hold': {
            'Total_Return': bh_return,
            'Annual_Return': bh_annual_return,
            'Volatility': bh_volatility,
            'Sharpe_Ratio': bh_sharpe,
            'Max_Drawdown': bh_max_drawdown
        },
        'Comparison': {
            'Return_Difference': strategy_return - bh_return,
            'Sharpe_Difference': strategy_sharpe - bh_sharpe,
            'Drawdown_Improvement': bh_max_drawdown - strategy_max_drawdown
        }
    }
    
    return backtest, results