import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

def load_and_prepare():
    """Load data and prepare for analysis"""
    print("="*70)
    print("ERROR ANALYSIS - WHERE DOES THE MODEL FAIL?")
    print("="*70)
    print("\nLoading data...")
    
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
    
    # Temporal split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df[split_idx:].reset_index(drop=True)
    
    # Train model
    print("Training model...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Add predictions and errors to test dataframe
    df_test['Predicted'] = predictions.astype(int)
    df_test['Error'] = (df_test['Attendance'] - df_test['Predicted']).astype(int)
    df_test['Abs_Error'] = np.abs(df_test['Error']).astype(int)
    df_test['Error_Pct'] = (df_test['Error'] / df_test['Attendance'] * 100).round(1)
    
    print(f"✓ {len(df_test)} test matches analyzed\n")
    
    return df_test

def analyze_worst_predictions(df):
    """Show worst predictions"""
    print("-"*70)
    print("TOP 10 WORST PREDICTIONS")
    print("-"*70)
    
    worst = df.nlargest(10, 'Abs_Error')[['Date', 'Home', 'Away', 'Attendance', 'Predicted', 'Error', 'Abs_Error']]  # ← ADICIONAR Abs_Error aqui
    
    print(f"\n{'Date':<12} {'Home':<20} {'Away':<20} {'Real':>8} {'Pred':>8} {'Error':>8}")
    print("-"*90)
    
    for _, row in worst.iterrows():
        print(f"{row['Date']:<12} {row['Home']:<20} {row['Away']:<20} "
              f"{row['Attendance']:>8,d} {row['Predicted']:>8,d} {row['Error']:>8,d}")
    
    print(f"\nMédia dos piores 10 erros: {worst['Abs_Error'].mean():,.0f} pessoas")

def analyze_by_team_type(df):
    """Analyze errors by team type"""
    print("\n" + "-"*70)
    print("ERRORS BY TEAM TYPE")
    print("-"*70)
    
    # Big3 vs Non-Big3
    big3_home = df[df['Home_Is_Big3'] == 1]
    non_big3_home = df[df['Home_Is_Big3'] == 0]
    
    print(f"\nBig3 Home Games ({len(big3_home)} matches):")
    print(f"  MAE:           {big3_home['Abs_Error'].mean():>10,.0f} pessoas")
    print(f"  Median Error:  {big3_home['Abs_Error'].median():>10,.0f} pessoas")
    print(f"  Max Error:     {big3_home['Abs_Error'].max():>10,.0f} pessoas")
    
    print(f"\nNon-Big3 Home Games ({len(non_big3_home)} matches):")
    print(f"  MAE:           {non_big3_home['Abs_Error'].mean():>10,.0f} pessoas")
    print(f"  Median Error:  {non_big3_home['Abs_Error'].median():>10,.0f} pessoas")
    print(f"  Max Error:     {non_big3_home['Abs_Error'].max():>10,.0f} pessoas")
    
    # Big games (Big3 vs Big3)
    big_games = df[(df['Home_Is_Big3'] == 1) & (df['Away_Is_Big3'] == 1)]
    
    print(f"\nBig Games (Big3 vs Big3) ({len(big_games)} matches):")
    print(f"  MAE:           {big_games['Abs_Error'].mean():>10,.0f} pessoas")
    print(f"  Median Error:  {big_games['Abs_Error'].median():>10,.0f} pessoas")

def analyze_overestimations(df):
    """Analyze where model overestimates"""
    print("\n" + "-"*70)
    print("OVERESTIMATIONS (Model predicts TOO HIGH)")
    print("-"*70)
    
    over = df[df['Error'] < -2000].sort_values('Error')
    
    print(f"\nTotal overestimations > 2k: {len(over)} matches\n")
    
    if len(over) > 0:
        print(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Real':>8} {'Pred':>8} {'Over':>8}")
        print("-"*90)
        
        for _, row in over.head(5).iterrows():
            print(f"{row['Date']:<12} {row['Home']:<20} {row['Away']:<20} "
                  f"{row['Attendance']:>8,d} {row['Predicted']:>8,d} {row['Error']:>8,d}")
        
        # Pattern analysis
        print(f"\nPatterns:")
        print(f"  Big3 home: {(over['Home_Is_Big3']==1).sum()} / {len(over)} ({(over['Home_Is_Big3']==1).sum()/len(over)*100:.0f}%)")
        print(f"  Big3 away: {(over['Away_Is_Big3']==1).sum()} / {len(over)} ({(over['Away_Is_Big3']==1).sum()/len(over)*100:.0f}%)")

def analyze_underestimations(df):
    """Analyze where model underestimates"""
    print("\n" + "-"*70)
    print("UNDERESTIMATIONS (Model predicts TOO LOW)")
    print("-"*70)
    
    under = df[df['Error'] > 2000].sort_values('Error', ascending=False)
    
    print(f"\nTotal underestimations > 2k: {len(under)} matches\n")
    
    if len(under) > 0:
        print(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Real':>8} {'Pred':>8} {'Under':>8}")
        print("-"*90)
        
        for _, row in under.head(5).iterrows():
            print(f"{row['Date']:<12} {row['Home']:<20} {row['Away']:<20} "
                  f"{row['Attendance']:>8,d} {row['Predicted']:>8,d} {row['Error']:>8,d}")
        
        # Pattern analysis
        print(f"\nPatterns:")
        print(f"  Big3 home: {(under['Home_Is_Big3']==1).sum()} / {len(under)} ({(under['Home_Is_Big3']==1).sum()/len(under)*100:.0f}%)")
        print(f"  Big3 away: {(under['Away_Is_Big3']==1).sum()} / {len(under)} ({(under['Away_Is_Big3']==1).sum()/len(under)*100:.0f}%)")

def analyze_by_round(df):
    """Analyze errors by round"""
    print("\n" + "-"*70)
    print("ERRORS BY SEASON PHASE")
    print("-"*70)
    
    first_half = df[df['Round_Num'] <= 17]
    second_half = df[df['Round_Num'] > 17]
    
    print(f"\nFirst Half (Rounds 1-17):")
    print(f"  MAE: {first_half['Abs_Error'].mean():>10,.0f} pessoas")
    
    print(f"\nSecond Half (Rounds 18-34):")
    print(f"  MAE: {second_half['Abs_Error'].mean():>10,.0f} pessoas")

def analyze_specific_teams(df):
    """Analyze errors for specific teams"""
    print("\n" + "-"*70)
    print("ERRORS BY SPECIFIC TEAMS (Home)")
    print("-"*70)
    
    team_errors = df.groupby('Home').agg({
        'Abs_Error': ['mean', 'count']
    }).round(0)
    
    team_errors.columns = ['MAE', 'Games']
    team_errors = team_errors[team_errors['Games'] >= 3]  # Min 3 games
    team_errors = team_errors.sort_values('MAE', ascending=False)
    
    print(f"\nTop 10 Hardest to Predict (Home):\n")
    print(f"{'Team':<25} {'MAE':>10} {'Games':>8}")
    print("-"*50)
    
    for team, row in team_errors.head(10).iterrows():
        print(f"{team:<25} {int(row['MAE']):>10,d} {int(row['Games']):>8d}")
    
    print(f"\nEasiest to Predict (Home):\n")
    print(f"{'Team':<25} {'MAE':>10} {'Games':>8}")
    print("-"*50)
    
    for team, row in team_errors.tail(5).iterrows():
        print(f"{team:<25} {int(row['MAE']):>10,d} {int(row['Games']):>8d}")

def summary_insights(df):
    """Provide summary insights"""
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    big3_mae = df[df['Home_Is_Big3']==1]['Abs_Error'].mean()
    non_big3_mae = df[df['Home_Is_Big3']==0]['Abs_Error'].mean()
    
    print(f"\n1. Big3 vs Non-Big3:")
    print(f"   Big3 MAE: {big3_mae:,.0f} | Non-Big3 MAE: {non_big3_mae:,.0f}")
    if big3_mae > non_big3_mae:
        print(f"   → Model struggles MORE with Big3 (harder to predict)")
    else:
        print(f"   → Model predicts Big3 better (more consistent)")
    
    print(f"\n2. Error Distribution:")
    within_1k = (df['Abs_Error'] <= 1000).sum()
    within_2k = (df['Abs_Error'] <= 2000).sum()
    within_5k = (df['Abs_Error'] <= 5000).sum()
    
    print(f"   Within 1k: {within_1k}/{len(df)} ({within_1k/len(df)*100:.1f}%)")
    print(f"   Within 2k: {within_2k}/{len(df)} ({within_2k/len(df)*100:.1f}%)")
    print(f"   Within 5k: {within_5k}/{len(df)} ({within_5k/len(df)*100:.1f}%)")
    
    print(f"\n3. Biggest Challenge:")
    worst_team = df.groupby('Home')['Abs_Error'].mean().idxmax()
    worst_mae = df.groupby('Home')['Abs_Error'].mean().max()
    print(f"   Hardest team to predict: {worst_team} (MAE: {worst_mae:,.0f})")

if __name__ == "__main__":
    
    # Load and analyze
    df_test = load_and_prepare()
    
    # Run analyses
    analyze_worst_predictions(df_test)
    analyze_by_team_type(df_test)
    analyze_overestimations(df_test)
    analyze_underestimations(df_test)
    analyze_by_round(df_test)
    analyze_specific_teams(df_test)
    summary_insights(df_test)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)