# scripts/analysis/machine_learning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def select_optimal_features(features_df, target_col):
    """Select the most informative features using various methods"""
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    
    # Prepare data
    X = features_df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col], errors='ignore')
    y = features_df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # 1. Linear correlation (F-test)
    f_selector = SelectKBest(f_regression, k=10)
    f_selector.fit(X, y)
    f_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': f_selector.scores_,
        'P_Value': f_selector.pvalues_
    }).sort_values('F_Score', ascending=False)
    
    # 2. Mutual Information (non-linear relationships)
    mi_selector = SelectKBest(mutual_info_regression, k=10)
    mi_selector.fit(X, y)
    mi_scores = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_selector.scores_
    }).sort_values('MI_Score', ascending=False)
    
    # 3. Feature Importance from Tree-based models
    from sklearn.ensemble import RandomForestRegressor
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance_scores = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'f_test': f_scores,
        'mutual_info': mi_scores,
        'random_forest': importance_scores,
        'recommended_features': importance_scores['Feature'][:10].tolist()
    }

def build_predictive_models(features_df, target_col, test_size=0.2):
    """Build and evaluate multiple ML models for price prediction"""
    # Prepare data
    X = features_df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col], errors='ignore')
    y = features_df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }
    
    # Results
    results = {}
    
    # Evaluate each model
    for name, model in models.items():
        # Metrics
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        # Cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        # Store results
        results[name] = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores)
        }
    
    # Identify best model
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    
    return {
        'model_results': results,
        'best_model': best_model[0],
        'best_metrics': best_model[1]
    }