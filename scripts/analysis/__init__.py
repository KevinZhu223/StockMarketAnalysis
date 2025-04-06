# scripts/analysis/__init__.py

# Import from your existing feature_engineering.py
from scripts.analysis.feature_engineering import add_tech_indicators, enhance_sentiment_features, combine_stock_sentiment

# Import from new modules
from .technical_analysis import calculate_advanced_indicators, detect_candlestick_patterns
from .statistical_analysis import test_market_efficiency, perform_correlation_analysis, perform_risk_analysis
from .trading_strategies import define_trading_strategies, backtest_strategy
from .machine_learning import select_optimal_features, build_predictive_models