import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train final model on ALL data and save it"""
    
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
    
    # Train
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # Save
    with open('model_attendance.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    return model, le, feature_cols

def predict_attendance(home, away, round_num, day_type,
                      home_avg, home_last3, matchup_avg,
                      home_is_big3, away_is_big3):
    """Make a prediction for a single match"""
    
    with open('model_attendance.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    features = pd.DataFrame([[
        home_avg,
        home_last3,
        matchup_avg,
        home_is_big3,
        away_is_big3,
        le.transform([day_type])[0],
        round_num
    ]], columns=[
        'Home_Avg_Attendance',
        'Home_Last3_Avg',
        'Matchup_Avg_Attendance',
        'Home_Is_Big3',
        'Away_Is_Big3',
        'Day_Type_Encoded',
        'Round_Num'
    ])
    
    return int(model.predict(features)[0])

def load_final_model():
    """Load trained model, label encoder and feature names for the dashboard."""
    with open('model_attendance.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    with open('feature_names.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    return model, le, feature_cols


if __name__ == "__main__":
    
    model, le, features = train_and_save_model()
    print("Model trained and saved successfully")