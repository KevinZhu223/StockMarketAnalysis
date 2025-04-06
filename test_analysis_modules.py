import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# At the top of test_analysis_modules.py
print("=== Starting test_analysis_modules.py ===")

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
            
def test_minimal_functionality():
    """A minimal test that doesn't depend on other modules"""
    print("Running minimal test...")
    
    # Create a simple DataFrame
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create a simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'])
    plt.title('Sample Data')
    plt.savefig('test_results/figures/minimal_test.png')
    print("Saved minimal test figure")
    
    # Save sample data
    data.to_csv('test_results/data/minimal_data.csv')
    print("Saved minimal test data")
    
    # Write to log
    logger.info("Minimal test completed successfully")
    
    return True

# Configure plotting
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Before creating directories
print("Creating test directories...")

# Create directories for test results
os.makedirs('test_results/figures', exist_ok=True)
os.makedirs('test_results/data', exist_ok=True)

# After creating directories
print(f"Directories created. Current working directory: {os.getcwd()}")

# Set up basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('analysis_test')

# Before loading data
print("About to load test data...")


# Load sample data (using a well-known stock like AAPL)
def load_test_data():
    """Load sample stock data for testing with proper error handling and data validation"""
    try:
        # Try to load from processed data if available
        stock_path = 'data/processed/AAPL_stock_clean.csv'
        if os.path.exists(stock_path):
            print(f"Found existing data file: {stock_path}")
            
            # Load data with explicit dtype specification
            df = pd.read_csv(stock_path, parse_dates=True)
            
            # Check if the data has a proper index
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
            # Debug the loaded data
            debug_data_structure(df, "Loaded Data")
            
            # Convert string columns to numeric if needed
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns and df[col].dtype == 'object':
                    print(f"Converting {col} from {df[col].dtype} to numeric")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded existing data from {stock_path}")
        else:
            print(f"File not found: {stock_path}, downloading fresh data")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname('test_results/data/AAPL_test_data.csv'), exist_ok=True)
            
            # Download fresh data
            import yfinance as yf
            df = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
            
            # Save for future use
            df.to_csv('test_results/data/AAPL_test_data.csv')
            logger.info("Downloaded fresh AAPL data")
            
        # Final validation
        debug_data_structure(df, "Processed Data")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal test data as fallback
        print("Creating minimal test data as fallback")
        dates = pd.date_range(start='2022-01-01', periods=100)
        df = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    # Basic data validation
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        # Add missing columns with default values if needed
        for col in missing_columns:
            df[col] = 0
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            # Create a default datetime index if conversion fails
            print("Failed to convert index to datetime, creating new index")
            df.index = pd.date_range(start='2022-01-01', periods=len(df))
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

# After loading data
print("Test data loaded successfully.")

# Before each test function
print("Starting technical analysis test...")

def test_technical_analysis():
    """Test the technical analysis module"""
    logger.info("Testing technical analysis module...")
    
    # Import the necessary functions
    from scripts.analysis.feature_engineering import add_tech_indicators
    from scripts.analysis.technical_analysis import calculate_advanced_indicators, detect_candlestick_patterns
    
    # Load test data
    stock_data = load_test_data()
    
    # Test basic technical indicators
    try:
        logger.info("Testing basic technical indicators...")
        basic_indicators = add_tech_indicators(stock_data)
        logger.info(f"Basic indicators added. New shape: {basic_indicators.shape}")
        
        # Verify key indicators were created
        expected_indicators = ['SMA_20', 'EMA_12', 'RSI_14', 'MACD']
        missing_indicators = [ind for ind in expected_indicators if ind not in basic_indicators.columns]
        
        if missing_indicators:
            logger.warning(f"Missing expected indicators: {missing_indicators}")
        else:
            logger.info("All expected basic indicators were created successfully")
        
        # Create visualization of basic indicators
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(basic_indicators.index, basic_indicators['Close'], label='Price')
        if 'SMA_20' in basic_indicators.columns:
            plt.plot(basic_indicators.index, basic_indicators['SMA_20'], label='SMA 20')
        if 'SMA_50' in basic_indicators.columns:
            plt.plot(basic_indicators.index, basic_indicators['SMA_50'], label='SMA 50')
        plt.legend()
        plt.title('Price and Moving Averages')
        
        plt.subplot(3, 1, 2)
        if 'RSI_14' in basic_indicators.columns:
            plt.plot(basic_indicators.index, basic_indicators['RSI_14'], label='RSI')
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
        plt.legend()
        plt.title('RSI')
        
        plt.subplot(3, 1, 3)
        if all(col in basic_indicators.columns for col in ['MACD', 'MACD_Signal']):
            plt.plot(basic_indicators.index, basic_indicators['MACD'], label='MACD')
            plt.plot(basic_indicators.index, basic_indicators['MACD_Signal'], label='Signal')
        plt.legend()
        plt.title('MACD')
        
        plt.tight_layout()
        plt.savefig('test_results/figures/basic_indicators.png')
        logger.info("Saved visualization of basic indicators")
    except Exception as e:
        logger.error(f"Error testing basic indicators: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test advanced technical indicators
    try:
        logger.info("Testing advanced technical indicators...")
        advanced_indicators = calculate_advanced_indicators(stock_data)
        logger.info(f"Advanced indicators added. New shape: {advanced_indicators.shape}")
        
        # Verify key advanced indicators were created
        expected_advanced = ['Tenkan_Sen', 'Kijun_Sen', 'MFI', 'Chaikin_Osc']
        missing_advanced = [ind for ind in expected_advanced if ind not in advanced_indicators.columns]
        
        if missing_advanced:
            logger.warning(f"Missing expected advanced indicators: {missing_advanced}")
        else:
            logger.info("All expected advanced indicators were created successfully")
        
        # Create visualization of advanced indicators
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(advanced_indicators.index, advanced_indicators['Close'], label='Price')
        if all(col in advanced_indicators.columns for col in ['Tenkan_Sen', 'Kijun_Sen']):
            plt.plot(advanced_indicators.index, advanced_indicators['Tenkan_Sen'], label='Tenkan Sen')
            plt.plot(advanced_indicators.index, advanced_indicators['Kijun_Sen'], label='Kijun Sen')
        plt.legend()
        plt.title('Ichimoku Cloud Components')
        
        plt.subplot(3, 1, 2)
        if 'MFI' in advanced_indicators.columns:
            plt.plot(advanced_indicators.index, advanced_indicators['MFI'], label='MFI')
            plt.axhline(y=80, color='r', linestyle='--')
            plt.axhline(y=20, color='g', linestyle='--')
        plt.legend()
        plt.title('Money Flow Index')
        
        plt.subplot(3, 1, 3)
        if 'Chaikin_Osc' in advanced_indicators.columns:
            plt.plot(advanced_indicators.index, advanced_indicators['Chaikin_Osc'], label='Chaikin Oscillator')
        plt.legend()
        plt.title('Chaikin Oscillator')
        
        plt.tight_layout()
        plt.savefig('test_results/figures/advanced_indicators.png')
        logger.info("Saved visualization of advanced indicators")
    except Exception as e:
        logger.error(f"Error testing advanced indicators: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test candlestick pattern detection
    try:
        logger.info("Testing candlestick pattern detection...")
        patterns_df = detect_candlestick_patterns(stock_data)
        logger.info(f"Candlestick patterns detected. New shape: {patterns_df.shape}")
        
        # Verify key patterns were identified
        expected_patterns = ['Doji', 'Hammer', 'ShootingStar', 'BullishEngulfing', 'BearishEngulfing']
        missing_patterns = [pat for pat in expected_patterns if pat not in patterns_df.columns]
        
        if missing_patterns:
            logger.warning(f"Missing expected patterns: {missing_patterns}")
        else:
            logger.info("All expected candlestick patterns were detected successfully")
        
        # Count occurrences of each pattern
        pattern_counts = {pat: patterns_df[pat].sum() for pat in expected_patterns if pat in patterns_df.columns}
        logger.info(f"Pattern occurrences: {pattern_counts}")
        
        # Create visualization of pattern occurrences
        plt.figure(figsize=(10, 6))
        plt.bar(pattern_counts.keys(), pattern_counts.values())
        plt.title('Candlestick Pattern Occurrences')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('test_results/figures/candlestick_patterns.png')
        logger.info("Saved visualization of candlestick patterns")
    except Exception as e:
        logger.error(f"Error testing candlestick patterns: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Technical analysis testing completed")
    
# Before each test function
print("Starting statistical analysis test...")

def test_statistical_analysis():
    """Test the statistical analysis module"""
    logger.info("Testing statistical analysis module...")
    
    # Import the necessary functions
    from scripts.analysis.feature_engineering import add_tech_indicators
    from scripts.analysis.statistical_analysis import test_market_efficiency, perform_correlation_analysis, perform_risk_analysis
    
    # Load and prepare test data
    stock_data = load_test_data()
    
    # Calculate basic indicators
    stock_with_indicators = add_tech_indicators(stock_data)
    
    # Add daily returns if not present
    if 'Daily_Return' not in stock_with_indicators.columns:
        stock_with_indicators['Daily_Return'] = stock_with_indicators['Close'].pct_change()
    
    # Test market efficiency
    try:
        logger.info("Testing market efficiency analysis...")
        returns = stock_with_indicators['Daily_Return'].dropna()
        efficiency_results = test_market_efficiency(returns)
        
        # Log results
        logger.info("Market efficiency test results:")
        for test_name, result in efficiency_results.items():
            logger.info(f"  {test_name}:")
            for key, value in result.items():
                logger.info(f"    {key}: {value}")
        
        # Create summary table
        efficiency_summary = pd.DataFrame({
            'Test': [],
            'Statistic': [],
            'p-value': [],
            'Interpretation': []
        })
        
        for test_name, result in efficiency_results.items():
            row = {
                'Test': test_name,
                'Statistic': result.get('statistic', 'N/A'),
                'p-value': result.get('p_value', 'N/A')
            }
            
            # Add interpretation
            if test_name == 'jarque_bera':
                row['Interpretation'] = 'Normal' if result.get('is_normal', False) else 'Not Normal'
            elif test_name == 'ljung_box':
                row['Interpretation'] = 'Independent' if result.get('is_independent', False) else 'Autocorrelated'
            elif test_name == 'runs_test':
                row['Interpretation'] = 'Random' if result.get('is_random', False) else 'Not Random'
            
            efficiency_summary = pd.concat([efficiency_summary, pd.DataFrame([row])], ignore_index=True)
        
        # Save results
        efficiency_summary.to_csv('test_results/data/market_efficiency.csv', index=False)
        logger.info("Saved market efficiency results")
    except Exception as e:
        logger.error(f"Error testing market efficiency: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test correlation analysis
    try:
        logger.info("Testing correlation analysis...")
        correlation_results = perform_correlation_analysis(stock_data, stock_with_indicators)
        
        # Log results
        logger.info("Correlation analysis results:")
        for period, result in correlation_results.items():
            logger.info(f"  {period}:")
            for i, feature in enumerate(result['top_features'].index[:5]):
                corr_value = result['correlation_values'][feature]
                logger.info(f"    {i+1}. {feature}: {corr_value:.4f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Select relevant columns for correlation
        corr_columns = ['Close', 'Volume', 'Daily_Return']
        corr_columns.extend([col for col in stock_with_indicators.columns 
                           if any(x in col for x in ['SMA', 'EMA', 'RSI', 'MACD'])][:10])
        
        correlation_matrix = stock_with_indicators[corr_columns].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Key Indicators')
        plt.tight_layout()
        plt.savefig('test_results/figures/correlation_heatmap.png')
        logger.info("Saved correlation heatmap")
    except Exception as e:
        logger.error(f"Error testing correlation analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test risk analysis
    try:
        logger.info("Testing risk analysis...")
        risk_metrics = perform_risk_analysis(stock_with_indicators)
        
        # Log results
        if 'basic' in risk_metrics:
            logger.info("Risk analysis results:")
            for metric, value in risk_metrics['basic'].items():
                logger.info(f"  {metric}: {value:.6f}")
        
        # Create risk metrics table
        if 'basic' in risk_metrics:
            risk_df = pd.DataFrame({
                'Metric': risk_metrics['basic'].keys(),
                'Value': risk_metrics['basic'].values()
            })
            
            # Save results
            risk_df.to_csv('test_results/data/risk_metrics.csv', index=False)
            logger.info("Saved risk metrics")
            
            # Create visualization of key risk metrics
            plt.figure(figsize=(10, 6))
            plt.bar(
                ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
                [
                    risk_metrics['basic']['annualized_return'] * 100,
                    risk_metrics['basic']['annualized_volatility'] * 100,
                    risk_metrics['basic']['sharpe_ratio'],
                    risk_metrics['basic']['sortino_ratio'],
                    risk_metrics['basic']['max_drawdown'] * 100
                ]
            )
            plt.title('Key Risk Metrics')
            plt.ylabel('Value (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('test_results/figures/risk_metrics.png')
            logger.info("Saved risk metrics visualization")
    except Exception as e:
        logger.error(f"Error testing risk analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Statistical analysis testing completed")
    
# Before each test function
print("Starting trading strategies analysis test...")

def test_trading_strategies():
    """Test the trading strategies module"""
    logger.info("Testing trading strategies module...")
    
    # Import the necessary functions
    from scripts.analysis.feature_engineering import add_tech_indicators
    from scripts.analysis.trading_strategies import define_trading_strategies, backtest_strategy
    
    # Load and prepare test data
    stock_data = load_test_data()
    
    # Calculate technical indicators
    stock_with_indicators = add_tech_indicators(stock_data)
    
    # Test strategy definition
    try:
        logger.info("Testing strategy definition...")
        strategies = define_trading_strategies(stock_with_indicators)
        
        # Log defined strategies
        logger.info(f"Defined {len(strategies)} strategies:")
        for name, strategy in strategies.items():
            logger.info(f"  {name}: {strategy['description']}")
        
        # Save strategy signals
        strategy_signals = pd.DataFrame(index=stock_with_indicators.index)
        strategy_signals['Close'] = stock_with_indicators['Close']
        
        for name, strategy in strategies.items():
            strategy_signals[name] = strategy['signal'].astype(int)
        
        strategy_signals.to_csv('test_results/data/strategy_signals.csv')
        logger.info("Saved strategy signals")
        
        # Visualize strategy signals
        plt.figure(figsize=(12, 8))
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(strategy_signals.index, strategy_signals['Close'])
        ax1.set_title('Price Chart')
        ax1.set_ylabel('Price ($)')
        
        # Plot signals
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, strategy_name in enumerate(strategies.keys()):
            if strategy_name in strategy_signals.columns:
                ax2.plot(
                    strategy_signals.index, 
                    strategy_signals[strategy_name], 
                    label=strategy_name,
                    color=colors[i % len(colors)],
                    alpha=0.7
                )
        
        ax2.set_title('Strategy Signals (1 = Buy, 0 = Sell/Hold)')
        ax2.set_ylabel('Signal')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('test_results/figures/strategy_signals.png')
        logger.info("Saved strategy signals visualization")
    except Exception as e:
        logger.error(f"Error testing strategy definition: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test strategy backtesting
    try:
        logger.info("Testing strategy backtesting...")
        
        # Test each strategy
        backtest_results = {}
        performance_metrics = {}
        
        for name, strategy in strategies.items():
            logger.info(f"Backtesting {name} strategy...")
            backtest_df, results = backtest_strategy(stock_with_indicators, strategy['signal'])
            
            backtest_results[name] = backtest_df
            performance_metrics[name] = results
            
            logger.info(f"  Total Return: {results['Strategy']['Total_Return']*100:.2f}%")
            logger.info(f"  Sharpe Ratio: {results['Strategy']['Sharpe_Ratio']:.4f}")
            logger.info(f"  Max Drawdown: {results['Strategy']['Max_Drawdown']*100:.2f}%")
            logger.info(f"  vs. Buy & Hold: {results['Comparison']['Return_Difference']*100:+.2f}%")
        
        # Create performance comparison dataframe
        performance_df = pd.DataFrame(columns=[
            'Strategy', 'Total Return (%)', 'Annual Return (%)', 
            'Sharpe Ratio', 'Max Drawdown (%)', 'vs. Buy & Hold (%)'
        ])
        
        # Add buy & hold as baseline
        first_strategy = list(performance_metrics.keys())[0]
        buy_hold = performance_metrics[first_strategy]['Buy_Hold']
        
        performance_df = pd.concat([performance_df, pd.DataFrame([{
            'Strategy': 'Buy & Hold',
            'Total Return (%)': buy_hold['Total_Return'] * 100,
            'Annual Return (%)': buy_hold['Annual_Return'] * 100,
            'Sharpe Ratio': buy_hold['Sharpe_Ratio'],
            'Max Drawdown (%)': buy_hold['Max_Drawdown'] * 100,
            'vs. Buy & Hold (%)': 0.0
        }])], ignore_index=True)
        
        # Add each strategy
        for name, metrics in performance_metrics.items():
            performance_df = pd.concat([performance_df, pd.DataFrame([{
                'Strategy': name,
                'Total Return (%)': metrics['Strategy']['Total_Return'] * 100,
                'Annual Return (%)': metrics['Strategy']['Annual_Return'] * 100,
                'Sharpe Ratio': metrics['Strategy']['Sharpe_Ratio'],
                'Max Drawdown (%)': metrics['Strategy']['Max_Drawdown'] * 100,
                'vs. Buy & Hold (%)': metrics['Comparison']['Return_Difference'] * 100
            }])], ignore_index=True)
        
        # Save performance comparison
        performance_df.to_csv('test_results/data/strategy_performance.csv', index=False)
        logger.info("Saved strategy performance comparison")
        
        # Visualize equity curves
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        plt.subplot(2, 1, 1)
        
        # Get first backtest dataframe for buy & hold
        first_backtest = list(backtest_results.values())[0]
        plt.plot(first_backtest.index, first_backtest['Buy_Hold_Equity'], label='Buy & Hold', color='black', linewidth=2)
        
        for name, backtest_df in backtest_results.items():
            plt.plot(backtest_df.index, backtest_df['Strategy_Equity'], label=name)
        
        plt.title('Strategy Equity Curves')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        
        plt.plot(first_backtest.index, first_backtest['Buy_Hold_Drawdown'] * 100, label='Buy & Hold', color='black', linewidth=2)
        
        for name, backtest_df in backtest_results.items():
            plt.plot(backtest_df.index, backtest_df['Strategy_Drawdown'] * 100, label=name)
        
        plt.title('Strategy Drawdowns')
        plt.ylabel('Drawdown (%)')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('test_results/figures/strategy_performance.png')
        logger.info("Saved strategy performance visualization")
    except Exception as e:
        logger.error(f"Error testing strategy backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Trading strategies testing completed")
    
# Before each test function
print("Starting machine learning analysis test...")

def test_machine_learning():
    """Test the machine learning module"""
    logger.info("Testing machine learning module...")
    
    # Import the necessary functions
    from scripts.analysis.feature_engineering import add_tech_indicators
    from scripts.analysis.machine_learning import select_optimal_features, build_predictive_models
    
    # Load and prepare test data
    stock_data = load_test_data()
    
    # Calculate technical indicators
    stock_with_indicators = add_tech_indicators(stock_data)
    
    # Add target variables if not present
    price_col = 'Close'
    
    if 'Daily_Return' not in stock_with_indicators.columns:
        stock_with_indicators['Daily_Return'] = stock_with_indicators[price_col].pct_change()
    
    if 'target_next_day_return' not in stock_with_indicators.columns:
        stock_with_indicators['target_next_day_return'] = stock_with_indicators['Daily_Return'].shift(-1)
        stock_with_indicators['target_next_day_direction'] = (stock_with_indicators['target_next_day_return'] > 0).astype(int)
        
        stock_with_indicators['target_3day_return'] = stock_with_indicators[price_col].pct_change(periods=3).shift(-3)
        stock_with_indicators['target_5day_return'] = stock_with_indicators[price_col].pct_change(periods=5).shift(-5)
        
        stock_with_indicators['target_3day_direction'] = (stock_with_indicators['target_3day_return'] > 0).astype(int)
        stock_with_indicators['target_5day_direction'] = (stock_with_indicators['target_5day_return'] > 0).astype(int)
    
    # Drop rows with NaN in target variables
    stock_with_indicators = stock_with_indicators.dropna(subset=['target_next_day_return'])
    
    # Test feature selection
    try:
        logger.info("Testing feature selection...")
        
        feature_selection_results = {}
        
        for target in ['target_next_day_return', 'target_3day_return', 'target_5day_return']:
            logger.info(f"Selecting features for {target}...")
            selection_results = select_optimal_features(stock_with_indicators, target)
            feature_selection_results[target] = selection_results
            
            # Log top features
            logger.info(f"Top features for {target}:")
            for feature in selection_results['recommended_features'][:5]:
                logger.info(f"  - {feature}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 10))
        
        for i, target in enumerate(['target_next_day_return', 'target_3day_return', 'target_5day_return']):
            plt.subplot(3, 1, i+1)
            
            # Get importance dataframe
            importance_df = feature_selection_results[target]['random_forest']
            
            # Sort and get top 10
            top_features = importance_df.sort_values('Importance', ascending=False).head(10)
            
            # Plot
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Feature Importance for {target}')
            plt.tight_layout()
        
        plt.savefig('test_results/figures/feature_importance.png')
        logger.info("Saved feature importance visualization")
    except Exception as e:
        logger.error(f"Error testing feature selection: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test predictive models
    try:
        logger.info("Testing predictive models...")
        
        model_results = {}
        
        for target in ['target_next_day_return', 'target_3day_return', 'target_5day_return']:
            logger.info(f"Building models for {target}...")
            ml_results = build_predictive_models(stock_with_indicators, target)
            model_results[target] = ml_results
            
            # Log model performance
            logger.info(f"Model performance for {target}:")
            logger.info(f"  Best model: {ml_results['best_model']}")
            for metric, value in ml_results['best_metrics'].items():
                logger.info(f"  {metric}: {value:.6f}")
        
        # Create model performance comparison dataframe
        performance_df = pd.DataFrame(columns=[
            'Target', 'Model', 'RMSE', 'MAE', 'R2'
        ])
        
        for target, ml_results in model_results.items():
            for model_name, metrics in ml_results['model_results'].items():
                performance_df = pd.concat([performance_df, pd.DataFrame([{
                    'Target': target,
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2']
                }])], ignore_index=True)
        
        # Save model performance comparison
        performance_df.to_csv('test_results/data/model_performance.csv', index=False)
        logger.info("Saved model performance comparison")
        
        # Visualize model performance
        plt.figure(figsize=(15, 10))
        
        for i, target in enumerate(['target_next_day_return', 'target_3day_return', 'target_5day_return']):
            plt.subplot(3, 1, i+1)
            
            # Filter for this target
            target_df = performance_df[performance_df['Target'] == target]
            
            # Sort by RMSE (lower is better)
            target_df = target_df.sort_values('RMSE')
            
            # Plot
            sns.barplot(x='RMSE', y='Model', data=target_df)
            plt.title(f'Model Performance for {target} (RMSE, lower is better)')
            plt.tight_layout()
        
        plt.savefig('test_results/figures/model_performance.png')
        logger.info("Saved model performance visualization")
    except Exception as e:
        logger.error(f"Error testing predictive models: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Machine learning testing completed")
    
# At the end of the file
print("=== Completed test_analysis_modules.py ===")

def main():
    """Main function to run all tests"""
    print("Starting all tests...")
       
    # Run minimal test first
    test_minimal_functionality()
       
    # Then run other tests
    test_technical_analysis()
    test_statistical_analysis()
    test_trading_strategies()
    test_machine_learning()
       
    print("All tests completed!")

if __name__ == "__main__":
       main()