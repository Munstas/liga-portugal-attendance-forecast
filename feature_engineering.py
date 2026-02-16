import pandas as pd
import numpy as np
from datetime import datetime

def load_clean_data():
    """Load clean dataset"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING - LIGA PORTUGAL")
    print("="*70)
    print("\nLoading clean data...")
    
    df = pd.read_csv('data/liga_portugal_clean.csv')
    print(f"✓ {len(df)} matches loaded")
    
    return df

def create_temporal_features(df):
    """Create time-based features"""
    print("\nCreating temporal features...")
    
    # Parse date
    df['Date_parsed'] = pd.to_datetime(df['Date'])
    
    # Extract components
    df['Month'] = df['Date_parsed'].dt.month
    df['Year'] = df['Date_parsed'].dt.year
    
    # Season
    def get_season(date):
        year = date.year
        month = date.month
        if month >= 8:
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"
    
    df['Season'] = df['Date_parsed'].apply(get_season)
    
    # Weekend flag
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Fri', 'Sat', 'Sun']).astype(int)
    
    # Day group
    df['Day_Group'] = df['Day_of_Week'].map({
        'Fri': 'weekend',
        'Sat': 'weekend',
        'Sun': 'weekend',
        'Mon': 'weekday',
        'Tue': 'rare',
        'Wed': 'rare',
        'Thu': 'rare'
    })
    
    # Round number
    df['Round_Num'] = pd.to_numeric(df['Round'], errors='coerce')
    
    # Season phase
    df['Season_Phase'] = (df['Round_Num'] > 17).astype(int)
    
    print("  ✓ Temporal features created")
    
    return df

def create_team_features(df):
    """Create team-based features"""
    print("\nCreating team features...")
    
    # Big 3 teams
    big_3 = ['Benfica', 'FC Porto', 'Sporting CP']
    df['Home_Is_Big3'] = df['Home'].isin(big_3).astype(int)
    df['Away_Is_Big3'] = df['Away'].isin(big_3).astype(int)
    
    # Derby
    df['Is_Derby'] = (df['Home_Is_Big3'] == 1) & (df['Away_Is_Big3'] == 1)
    df['Is_Derby'] = df['Is_Derby'].astype(int)
    
    # Big vs Small
    df['Big_Home_Small_Away'] = (df['Home_Is_Big3'] == 1) & (df['Away_Is_Big3'] == 0)
    df['Big_Home_Small_Away'] = df['Big_Home_Small_Away'].astype(int)
    
    print("  ✓ Team features created")
    
    return df

def create_historical_features(df):
    """Create historical average features"""
    print("\nCreating historical features...")
    
    # Sort by date
    df = df.sort_values('Date_parsed').reset_index(drop=True)
    
    # Home team historical average
    df['Home_Avg_Attendance'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Away team effect
    away_avg = df.groupby('Away')['Attendance'].mean().to_dict()
    df['Away_Team_Draw'] = df['Away'].map(away_avg)
    
    # Fill NaN (fix warning)
    overall_mean = df['Attendance'].mean()
    df['Home_Avg_Attendance'] = df['Home_Avg_Attendance'].fillna(overall_mean)
    df['Away_Team_Draw'] = df['Away_Team_Draw'].fillna(overall_mean)
    
    # Recent form
    df['Home_Last3_Avg'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    df['Home_Last3_Avg'] = df['Home_Last3_Avg'].fillna(overall_mean)
    
    print("  ✓ Historical features created")
    
    return df

def create_matchup_features(df):
    """Create specific matchup features"""
    print("\nCreating matchup features...")
    
    # Matchup identifier
    df['Matchup'] = df['Home'] + ' vs ' + df['Away']
    
    # Historical average for matchup
    matchup_avg = df.groupby('Matchup')['Attendance'].mean().to_dict()
    df['Matchup_Historical_Avg'] = df['Matchup'].map(matchup_avg)
    
    # Matchup count
    df['Matchup_Count'] = df.groupby('Matchup').cumcount()
    
    print("  ✓ Matchup features created")
    
    return df

def select_final_features(df):
    """Select and order final features"""
    print("\nSelecting final features...")
    
    feature_cols = [
        # Target
        'Attendance',
        
        # Identifiers
        'Date',
        'Season',
        'Home',
        'Away',
        'Round',
        
        # Temporal
        'Month',
        'Is_Weekend',
        'Day_Group',
        'Round_Num',
        'Season_Phase',
        
        # Team
        'Home_Is_Big3',
        'Away_Is_Big3',
        'Is_Derby',
        'Big_Home_Small_Away',
        
        # Historical
        'Home_Avg_Attendance',
        'Away_Team_Draw',
        'Home_Last3_Avg',
        'Matchup_Historical_Avg',
        'Matchup_Count'
    ]
    
    df_final = df[feature_cols].copy()
    
    print(f"  ✓ {len(feature_cols)} columns selected")
    
    return df_final

def show_summary(df):
    """Show summary"""
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    
    print(f"\nDataset: {len(df)} matches")
    print(f"Features: 14 (excluding target + identifiers)")
    
    print(f"\nTemporal:")
    print(f"  • Weekend games: {df['Is_Weekend'].sum()} ({df['Is_Weekend'].sum()/len(df)*100:.1f}%)")
    print(f"  • Months covered: {df['Month'].nunique()}")
    
    print(f"\nTeams:")
    print(f"  • Big3 home games: {df['Home_Is_Big3'].sum()}")
    print(f"  • Derbies: {df['Is_Derby'].sum()}")
    
    print(f"\nTarget (Attendance):")
    print(f"  • Mean: {df['Attendance'].mean():,.0f}")
    print(f"  • Range: {df['Attendance'].min():,d} - {df['Attendance'].max():,d}")

if __name__ == "__main__":
    
    # Load
    df = load_clean_data()
    
    # Create features
    df = create_temporal_features(df)
    df = create_team_features(df)
    df = create_historical_features(df)
    df = create_matchup_features(df)
    
    # Select
    df_final = select_final_features(df)
    
    # Summary
    show_summary(df_final)
    
    # Save
    output_file = 'data/liga_portugal_features.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✓ Saved: {output_file}")