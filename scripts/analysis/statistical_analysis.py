import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

def test_market_efficiency(returns):
    results = {}
    
    #Normality test (Jarque-Bera)
    jb_test = stats.jarque_bera(returns)
    results['jarque_bera'] = {
        'statistic': jb_test[0],
        'p_value': jb_test[1],
        'is_normal': jb_test[1] > .05
    }
    
    #Autocorrelation Test (Ljung-Box)
    lb_test = sm.stats.acorr_ljungbox(returns, lags=[10])
    results['ljung_box'] = {
        'statistic': lb_test.iloc[0, 0],
        'p_value': lb_test.iloc[0, 1],
        'is_independent': lb_test.iloc[0, 1] > 0.05
    }
    
    #Runs test for Randomness
    # Count runs (sequences of consecutive positive or negative returns)
    pos_neg = np.sign(returns)
    runs = len([i for i in range(1, len(pos_neg)) if pos_neg[i] != pos_neg[i-1]]) + 1
    n_pos = sum(pos_neg > 0)
    n_neg = sum(pos_neg < 0)
    
    # Expected number of runs
    exp_runs = ((2 * n_pos * n_neg) / (n_pos + n_neg)) + 1
    std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / 
                      ((n_pos + n_neg)**2 * (n_pos + n_neg - 1)))
    
    z_stat = (runs - exp_runs) / std_runs
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    results['runs_test'] = {
        'runs': runs,
        'expected_runs': exp_runs,
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_random': p_value > 0.05
    }
    
    return results

def perform_correlation_analysis(stock_df, features_df):
    analysis_df = features_df.copy()
    
    for period in [1,3,5]:
        analysis_df[f'future_return_{period}d'] = stock_df['Close'].pct_change(period).shift(-period)
        
    correlation_results = {}
    
    for period in [1,3,5]:
        future_return = f'future_return_{period}d'
        correlations = analysis_df.select_dtypes(include=['float64', 'int64']).corrwith(analysis_df[future_return])
        
        #Sort by absolute correlation values
        sorted_correlations = correlations.abs().sort_values(ascending=False)
        
        #Store top 10 correlations (excluding the future return)
        top_features = sorted_correlations[sorted_correlations.index != future_return][:10]
        
        correlation_results[f'{period}_day'] = {
            'top_features': top_features,
            'correlation_values': {feature: correlations[feature] for feature in top_features.index}
            
        }
        
    return correlation_results

def perform_risk_analysis(features_df, returns_column='Daily_Return'):
    if returns_column not in features_df.columns:
        raise ValueError(f"Returns column '{returns_column}' not found in data")
    
    returns = features_df[returns_column].dropna()
    
    risk_metrics = {}
    
    # 1. Basic Risk Metrics
    # ---------------------
    # Trading days per year (standard assumption)
    trading_days = 252
    
    # Annualized metrics
    annualized_return = returns.mean() * trading_days
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Sortino ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days)
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Value at Risk (VaR) - 95% and 99% confidence levels
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Conditional VaR (CVaR) / Expected Shortfall
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Store basic metrics
    risk_metrics['basic'] = {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99
    }
    
    return risk_metrics
