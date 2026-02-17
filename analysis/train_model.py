import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data():
    """Load feature data"""
    print("="*70)
    print("MODEL TRAINING - LIGA PORTUGAL ATTENDANCE PREDICTION")
    print("="*70)
    print("\nLoading data...")
    
    df = pd.read_csv('data/liga_portugal_features.csv')
    print(f"✓ {len(df)} matches loaded")
    
    return df

def prepare_data(df):
    """Prepare features and target"""
    print("\nPreparing data...")
    
    # Encode categorical variable (Day_Type)
    le = LabelEncoder()
    df['Day_Type_Encoded'] = le.fit_transform(df['Day_Type'])
    
    # Features for modeling
    feature_cols = [
        'Home_Avg_Attendance',
        'Home_Last3_Avg',
        'Matchup_Avg_Attendance',
        'Home_Is_Big3',
        'Away_Is_Big3',
        'Day_Type_Encoded',
        'Round_Num'
    ]
    
    # Separate features and target
    X = df[feature_cols]
    y = df['Attendance']
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Samples: {X.shape[0]}")
    print(f"\nFeature names:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    return X, y, feature_cols

def split_data(X, y, test_size=0.2):
    """Split data into train and test (temporal split)"""
    print(f"\nSplitting data (test size: {test_size*100:.0f}%)...")
    
    # Temporal split (train on older games, test on recent)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✓ Train set: {len(X_train)} matches")
    print(f"✓ Test set:  {len(X_test)} matches")
    
    return X_train, X_test, y_train, y_test

def train_baseline(y_train, y_test):
    """Baseline model: predict mean attendance"""
    print("\n" + "-"*70)
    print("BASELINE MODEL (Mean Predictor)")
    print("-"*70)
    
    # Predict mean for all test samples
    baseline_pred = np.full(len(y_test), y_train.mean())
    
    mae = mean_absolute_error(y_test, baseline_pred)
    rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    r2 = r2_score(y_test, baseline_pred)
    
    print(f"\nMetrics:")
    print(f"  MAE:  {mae:>10,.0f} pessoas")
    print(f"  RMSE: {rmse:>10,.0f} pessoas")
    print(f"  R²:   {r2:>10.3f}")
    
    return {'name': 'Baseline', 'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': baseline_pred}

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n" + "-"*70)
    print("RANDOM FOREST")
    print("-"*70)
    print("\nTraining...")
    
    # Train model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nMetrics:")
    print(f"  MAE (train): {mae_train:>10,.0f} pessoas")
    print(f"  MAE (test):  {mae_test:>10,.0f} pessoas")
    print(f"  RMSE:        {rmse_test:>10,.0f} pessoas")
    print(f"  R²:          {r2_test:>10.3f}")
    
    return {
        'name': 'Random Forest',
        'model': rf,
        'mae': mae_test,
        'rmse': rmse_test,
        'r2': r2_test,
        'predictions': y_pred_test
    }

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "-"*70)
    print("GRADIENT BOOSTING")
    print("-"*70)
    print("\nTraining...")
    
    # Train model
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    gb.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nMetrics:")
    print(f"  MAE (train): {mae_train:>10,.0f} pessoas")
    print(f"  MAE (test):  {mae_test:>10,.0f} pessoas")
    print(f"  RMSE:        {rmse_test:>10,.0f} pessoas")
    print(f"  R²:          {r2_test:>10.3f}")
    
    return {
        'name': 'Gradient Boosting',
        'model': gb,
        'mae': mae_test,
        'rmse': rmse_test,
        'r2': r2_test,
        'predictions': y_pred_test
    }

def show_feature_importance(model, feature_names):
    """Show feature importance"""
    print("\nFeature Importance:")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:<25} {importances[idx]:.3f}")

def compare_models(results):
    """Compare all models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Sort by MAE (lower is better)
    results_sorted = sorted(results, key=lambda x: x['mae'])
    
    print(f"\n{'Model':<20} {'MAE':<15} {'RMSE':<15} {'R²':<10}")
    print("-"*70)
    
    for result in results_sorted:
        print(f"{result['name']:<20} {result['mae']:>10,.0f}     {result['rmse']:>10,.0f}     {result['r2']:>8.3f}")
    
    # Best model
    best = results_sorted[0]
    print(f"\n🏆 Best Model: {best['name']}")
    print(f"   MAE: {best['mae']:,.0f} pessoas")
    print(f"   → Em média, erra por {best['mae']:,.0f} pessoas")

def analyze_errors(y_test, predictions, model_name):
    """Analyze prediction errors"""
    errors = y_test.values - predictions
    abs_errors = np.abs(errors)
    
    print(f"\n{model_name} - Error Analysis:")
    print(f"  Mean error:       {errors.mean():>10,.0f} pessoas")
    print(f"  Median abs error: {np.median(abs_errors):>10,.0f} pessoas")
    print(f"  Max error:        {abs_errors.max():>10,.0f} pessoas")
    print(f"  % within 2k:      {(abs_errors <= 2000).sum() / len(abs_errors) * 100:>10.1f}%")
    print(f"  % within 5k:      {(abs_errors <= 5000).sum() / len(abs_errors) * 100:>10.1f}%")

if __name__ == "__main__":
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_names = prepare_data(df)
    
    # Split data (temporal)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Train models
    results = []
    
    # 1. Baseline
    baseline_result = train_baseline(y_train, y_test)
    results.append(baseline_result)
    
    # 2. Random Forest
    rf_result = train_random_forest(X_train, X_test, y_train, y_test)
    results.append(rf_result)
    show_feature_importance(rf_result['model'], feature_names)
    analyze_errors(y_test, rf_result['predictions'], 'Random Forest')
    
    # 3. Gradient Boosting
    gb_result = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results.append(gb_result)
    show_feature_importance(gb_result['model'], feature_names)
    analyze_errors(y_test, gb_result['predictions'], 'Gradient Boosting')
    
    # Compare models
    compare_models(results)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\n✓ Models trained and evaluated")
    print("✓ Ready for predictions!")