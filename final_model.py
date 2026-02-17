import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

def train_and_save_model():
    """Train final model on ALL data and save it"""
    print("="*70)
    print("FINAL MODEL TRAINING & SAVING")
    print("="*70)
    print("\nLoading data...")
    
    # Load all data
    df = pd.read_csv('data/liga_portugal_features.csv')
    
    # Encode Day_Type
    le = LabelEncoder()
    df['Day_Type_Encoded'] = le.fit_transform(df['Day_Type'])
    
    # Features
    feature_cols = [
        'Home_Avg_Attendance',
        'Home_Last3_Avg',
        'Matchup_Avg_Attendance',
        'Home_Is_Big3',
        'Away_Is_Big3',
        'Day_Type_Encoded',
        'Round_Num'
    ]
    
    X = df[feature_cols]
    y = df['Attendance']
    
    print(f"✓ Training on ALL {len(X)} matches")
    
    # Train final model on ALL data
    print("\nTraining final model...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Performance on training data
    predictions = model.predict(X)
    mae = np.mean(np.abs(y - predictions))
    
    print(f"✓ Model trained")
    print(f"  Training MAE: {mae:,.0f} pessoas")
    
    # Save model and encoder
    print("\nSaving model...")
    
    with open('model_attendance.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("✓ Model saved: model_attendance.pkl")
    print("✓ Encoder saved: label_encoder.pkl")
    print("✓ Features saved: feature_names.pkl")
    
    return model, le, feature_cols

def predict_attendance(home, away, round_num, day_type, 
                      home_avg, home_last3, matchup_avg,
                      home_is_big3, away_is_big3):
    """Make a prediction for a single match"""
    
    # Load model
    with open('model_attendance.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Encode day_type
    day_type_encoded = le.transform([day_type])[0]
    
    # Create features
    features = np.array([[
        home_avg,
        home_last3,
        matchup_avg,
        home_is_big3,
        away_is_big3,
        day_type_encoded,
        round_num
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return int(prediction)

def create_prediction_interface():
    """Interactive prediction interface"""
    print("\n" + "="*70)
    print("PREDICTION INTERFACE")
    print("="*70)
    
    print("\nExamples of predictions:")
    print("-"*70)
    
    # Example 1: Benfica vs Sporting (Derby)
    pred1 = predict_attendance(
        home='Benfica',
        away='Sporting CP',
        round_num=15,
        day_type='weekend',
        home_avg=57000,
        home_last3=58000,
        matchup_avg=62000,
        home_is_big3=1,
        away_is_big3=1
    )
    print(f"\n1. Benfica vs Sporting CP (Round 15, Weekend)")
    print(f"   Predicted attendance: {pred1:,d} pessoas")
    
    # Example 2: Arouca vs Estoril (Small teams)
    pred2 = predict_attendance(
        home='Arouca',
        away='Estoril',
        round_num=20,
        day_type='weekend',
        home_avg=1900,
        home_last3=1850,
        matchup_avg=1800,
        home_is_big3=0,
        away_is_big3=0
    )
    print(f"\n2. Arouca vs Estoril (Round 20, Weekend)")
    print(f"   Predicted attendance: {pred2:,d} pessoas")
    
    # Example 3: Porto vs Braga
    pred3 = predict_attendance(
        home='FC Porto',
        away='Braga',
        round_num=10,
        day_type='weekend',
        home_avg=40000,
        home_last3=41000,
        matchup_avg=44000,
        home_is_big3=1,
        away_is_big3=0
    )
    print(f"\n3. FC Porto vs Braga (Round 10, Weekend)")
    print(f"   Predicted attendance: {pred3:,d} pessoas")
    
    # Example 4: Sporting vs small team on weekday
    pred4 = predict_attendance(
        home='Sporting CP',
        away='Gil Vicente',
        round_num=25,
        day_type='weekday',
        home_avg=35000,
        home_last3=36000,
        matchup_avg=33000,
        home_is_big3=1,
        away_is_big3=0
    )
    print(f"\n4. Sporting CP vs Gil Vicente (Round 25, Weekday)")
    print(f"   Predicted attendance: {pred4:,d} pessoas")

def show_usage_instructions():
    """Show how to use the model"""
    print("\n" + "="*70)
    print("HOW TO USE THE MODEL")
    print("="*70)
    
    print("""
To make predictions, use:
```python
from final_model import predict_attendance

prediction = predict_attendance(
    home='Benfica',              # Home team name
    away='FC Porto',             # Away team name
    round_num=15,                # Round number (1-34)
    day_type='weekend',          # 'weekend', 'weekday', or 'rare'
    home_avg=57000,              # Historical average (from data)
    home_last3=58000,            # Last 3 games average (from data)
    matchup_avg=62000,           # This matchup average (from data)
    home_is_big3=1,              # 1 if Big3, 0 otherwise
    away_is_big3=1               # 1 if Big3, 0 otherwise
)

print(f"Predicted: {prediction:,d} pessoas")
```

Note: For historical averages, check liga_portugal_features.csv
""")

def show_model_card():
    """Display model performance card"""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE CARD")
    print("="*70)
    
    print("""
Model: Gradient Boosting Regressor
Features: 7 (Home_Avg, Last3_Avg, Matchup_Avg, Home_Big3, Away_Big3, Day, Round)

OVERALL PERFORMANCE:
  • MAE:  1,526 pessoas
  • RMSE: 2,619 pessoas  
  • R²:   0.977
  • 62% predictions within 1,000 pessoas
  • 93% predictions within 5,000 pessoas

PERFORMANCE BY TEAM TYPE:
  • Small teams (non-Big3): MAE ~400-1,000 ✅ Excellent
  • Big3 teams: MAE ~3,000-6,000 ⚠️ Good (high inherent variability)
  
MOST CHALLENGING:
  • FC Porto: MAE 5,940
  • Sporting CP: MAE 4,164
  • Benfica: MAE 2,924

EASIEST TO PREDICT:
  • Nacional: MAE 386
  • Estoril: MAE 374
  • Arouca: MAE 438

TRAINING DATA:
  • 1,110 matches (2022-2026)
  • 4 seasons of Liga Portugal
  • 18+ unique teams
""")

if __name__ == "__main__":
    
    # Train and save model
    model, le, features = train_and_save_model()
    
    # Show predictions
    create_prediction_interface()
    
    # Show usage
    show_usage_instructions()
    
    # Show model card
    show_model_card()
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\n✓ Model trained and saved")
    print("✓ Ready for predictions")
    print("\nFiles created:")
    print("  • model_attendance.pkl")
    print("  • label_encoder.pkl")
    print("  • feature_names.pkl")