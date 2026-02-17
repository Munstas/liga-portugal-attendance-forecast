import pandas as pd
import pickle
import numpy as np
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load trained model"""
    with open('model_attendance.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    return model, le

def get_upcoming_matches():
    """Get upcoming matches from scraped data (future only)"""
    # Load original scraped data
    df = pd.read_csv('data/liga_portugal_fotmob.csv')
    
    # Parse dates
    df['Date_parsed'] = pd.to_datetime(df['Date'])
    today = pd.Timestamp.now()
    
    # Filter: ONLY future matches
    upcoming = df[df['Date_parsed'] > today]
    
    # Sort by date
    upcoming = upcoming.sort_values('Date_parsed')
    
    return upcoming

def get_team_stats(team_name):
    """Get historical stats for a team"""
    df = pd.read_csv('data/liga_portugal_features.csv')
    
    # Get most recent home games
    home_games = df[df['Home'] == team_name].tail(10)
    
    if len(home_games) == 0:
        # Team not in historical data
        return {
            'home_avg': 11951,
            'home_last3': 11951
        }
    
    # Most recent stats
    home_avg = home_games['Home_Avg_Attendance'].iloc[-1]
    home_last3 = home_games['Home_Last3_Avg'].iloc[-1]
    
    return {
        'home_avg': int(home_avg),
        'home_last3': int(home_last3)
    }

def get_matchup_stats(home, away):
    """Get historical stats for this specific matchup"""
    df = pd.read_csv('data/liga_portugal_features.csv')
    
    # Find previous games
    matchup = df[(df['Home'] == home) & (df['Away'] == away)]
    
    if len(matchup) == 0:
        # No previous matchup
        stats = get_team_stats(home)
        return stats['home_avg']
    
    matchup_avg = matchup['Matchup_Avg_Attendance'].iloc[-1]
    return int(matchup_avg)

def is_big3(team_name):
    """Check if team is Big3"""
    big3 = ['Benfica', 'FC Porto', 'Sporting CP']
    return 1 if team_name in big3 else 0

def predict_all_upcoming():
    """Predict all upcoming matches and save to CSV"""
    
    # Get upcoming matches
    upcoming = get_upcoming_matches()
    
    if len(upcoming) == 0:
        print("No upcoming matches found.")
        return
    
    # Load model
    model, le = load_model()
    
    predictions = []
    
    for idx, match in upcoming.iterrows():
        home = match['Home']
        away = match['Away']
        date = match['Date']
        round_num = int(match['Round'])
        day_of_week = match['Day_of_Week']
        
        # Map day to day_type
        if day_of_week in ['Fri', 'Sat', 'Sun']:
            day_type = 'weekend'
        elif day_of_week in ['Mon']:
            day_type = 'weekday'
        else:
            day_type = 'rare'
        
        # Get stats
        home_stats = get_team_stats(home)
        matchup_avg = get_matchup_stats(home, away)
        
        # Create features
        day_encoded = le.transform([day_type])[0]
        
        features = pd.DataFrame([[
            home_stats['home_avg'],
            home_stats['home_last3'],
            matchup_avg,
            is_big3(home),
            is_big3(away),
            day_encoded,
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
        
        # Predict
        prediction = int(model.predict(features)[0])
        
        # Confidence interval
        if is_big3(home):
            error_margin = 4000
        else:
            error_margin = 1000
        
        predictions.append({
            'Date': date,
            'Home': home,
            'Away': away,
            'Round': round_num,
            'Predicted_Attendance': prediction,
            'Lower_Bound': prediction - error_margin,
            'Upper_Bound': prediction + error_margin
        })
    
    # Save to CSV in data folder
    df_predictions = pd.DataFrame(predictions)
    output_path = 'data/predictions.csv'
    df_predictions.to_csv(output_path, index=False)
    
    print(f"✓ Predicted {len(predictions)} upcoming matches")
    print(f"✓ Saved to: {output_path}")

def predict_custom():
    """Interactive custom prediction"""
    print("\nCustom Match Prediction")
    print("-" * 50)
    
    home = input("Home team: ")
    away = input("Away team: ")
    round_num = int(input("Round (1-34): "))
    day_type = input("Day (weekend/weekday/rare): ")
    
    # Load model
    model, le = load_model()
    
    # Get stats
    home_stats = get_team_stats(home)
    matchup_avg = get_matchup_stats(home, away)
    
    # Create features
    day_encoded = le.transform([day_type])[0]
    
    features = pd.DataFrame([[
        home_stats['home_avg'],
        home_stats['home_last3'],
        matchup_avg,
        is_big3(home),
        is_big3(away),
        day_encoded,
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
    
    # Predict
    prediction = int(model.predict(features)[0])
    
    # Show result
    print(f"\n{home} vs {away}")
    print(f"Predicted: {prediction:,d} pessoas")
    
    if is_big3(home):
        print(f"Range: {prediction-4000:,d} - {prediction+4000:,d} (±4k)")
    else:
        print(f"Range: {prediction-1000:,d} - {prediction+1000:,d} (±1k)")

def show_teams():
    """Show available teams"""
    df = pd.read_csv('data/liga_portugal_features.csv')
    teams = sorted(df['Home'].unique())
    
    print("\nAvailable Teams:")
    print("-" * 50)
    
    for i, team in enumerate(teams, 1):
        games = len(df[df['Home'] == team])
        avg = df[df['Home'] == team]['Attendance'].mean()
        print(f"{i:2d}. {team:<25} {games:2d} games | avg: {avg:>7,.0f}")

if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'custom':
            predict_custom()
        elif cmd == 'teams':
            show_teams()
        else:
            print("Unknown command")
            print("Usage: python predict.py [custom|teams]")
    else:
        # Default: predict all upcoming
        predict_all_upcoming()