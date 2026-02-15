import pandas as pd
import numpy as np
from datetime import datetime

def load_clean_data():
    """Load clean dataset"""
    print("="*70)
    print("LOADING CLEAN DATA")
    print("="*70)
    
    df = pd.read_csv('data/liga_portugal_clean.csv')
    print(f"Total matches: {len(df)}")
    
    return df

def create_temporal_features(df):
    """Create time-based features"""
    print("\n" + "="*70)
    print("CREATING TEMPORAL FEATURES")
    print("="*70)
    
    # Parse date
    df['Date_parsed'] = pd.to_datetime(df['Date'])
    
    # Extract components
    df['Month'] = df['Date_parsed'].dt.month
    df['Year'] = df['Date_parsed'].dt.year
    
    # Season
    def get_season(date):
        year = date.year
        month = date.month
        if month >= 8:  # Aug-Dec
            return f"{year}-{year+1}"
        else:  # Jan-May
            return f"{year-1}-{year}"
    
    df['Season'] = df['Date_parsed'].apply(get_season)
    
    # Weekend flag (Fri, Sat, Sun have most games)
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Fri', 'Sat', 'Sun']).astype(int)
    
    # Day group (ignore Tue/Wed/Thu - too few samples)
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
    
    # Season phase (first half vs second half)
    df['Season_Phase'] = (df['Round_Num'] > 17).astype(int)  # 0 = first half, 1 = second half
    
    print(f"✓ Created temporal features")
    
    return df

def create_team_features(df):
    """Create team-based features"""
    print("\n" + "="*70)
    print("CREATING TEAM FEATURES")
    print("="*70)
    
    # Big 3 teams
    big_3 = ['Benfica', 'FC Porto', 'Sporting CP']
    df['Home_Is_Big3'] = df['Home'].isin(big_3).astype(int)
    df['Away_Is_Big3'] = df['Away'].isin(big_3).astype(int)
    
    # Derby (Big3 vs Big3)
    df['Is_Derby'] = (df['Home_Is_Big3'] == 1) & (df['Away_Is_Big3'] == 1)
    df['Is_Derby'] = df['Is_Derby'].astype(int)
    
    # Big vs Small (Big3 home vs small team)
    df['Big_Home_Small_Away'] = (df['Home_Is_Big3'] == 1) & (df['Away_Is_Big3'] == 0)
    df['Big_Home_Small_Away'] = df['Big_Home_Small_Away'].astype(int)
    
    print(f"✓ Created team features")
    
    return df

def create_historical_features(df):
    """Create historical average features"""
    print("\n" + "="*70)
    print("CREATING HISTORICAL FEATURES")
    print("="*70)
    
    # Sort by date
    df = df.sort_values('Date_parsed').reset_index(drop=True)
    
    # Home team historical average (expanding mean - only past games)
    df['Home_Avg_Attendance'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.expanding().mean().shift(1)  # shift(1) = exclude current game
    )
    
    # Away team effect (how much attendance they attract)
    # Calculate average attendance when this team plays AWAY
    away_avg = df.groupby('Away')['Attendance'].mean().to_dict()
    df['Away_Team_Draw'] = df['Away'].map(away_avg)
    
    # Fill NaN (first games of each team) with overall mean
    overall_mean = df['Attendance'].mean()
    df['Home_Avg_Attendance'].fillna(overall_mean, inplace=True)
    df['Away_Team_Draw'].fillna(overall_mean, inplace=True)
    
    # Recent form (last 3 home games average)
    df['Home_Last3_Avg'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    df['Home_Last3_Avg'].fillna(overall_mean, inplace=True)
    
    print(f"✓ Created historical features")
    
    return df

def create_matchup_features(df):
    """Create specific matchup features"""
    print("\n" + "="*70)
    print("CREATING MATCHUP FEATURES")
    print("="*70)
    
    # Create matchup identifier
    df['Matchup'] = df['Home'] + ' vs ' + df['Away']
    
    # Historical average for this specific matchup
    matchup_avg = df.groupby('Matchup')['Attendance'].mean().to_dict()
    df['Matchup_Historical_Avg'] = df['Matchup'].map(matchup_avg)
    
    # Count of times this matchup happened before
    df['Matchup_Count'] = df.groupby('Matchup').cumcount()
    
    print(f"✓ Created matchup features")
    
    return df

def select_final_features(df):
    """Select and order final features"""
    print("\n" + "="*70)
    print("SELECTING FINAL FEATURES")
    print("="*70)
    
    # Features for modeling
    feature_cols = [
        # Target
        'Attendance',
        
        # Identifiers (not used in model but useful)
        'Date',
        'Season',
        'Home',
        'Away',
        'Round',
        
        # Temporal features
        'Month',
        'Is_Weekend',
        'Day_Group',
        'Round_Num',
        'Season_Phase',
        
        # Team features
        'Home_Is_Big3',
        'Away_Is_Big3',
        'Is_Derby',
        'Big_Home_Small_Away',
        
        # Historical features
        'Home_Avg_Attendance',
        'Away_Team_Draw',
        'Home_Last3_Avg',
        'Matchup_Historical_Avg',
        'Matchup_Count'
    ]
    
    df_final = df[feature_cols].copy()
    
    print(f"✓ Selected {len(feature_cols)} columns")
    print(f"✓ Features: {len(feature_cols) - 6} (excluding target + identifiers)")
    
    return df_final

def show_feature_summary(df):
    """Show summary of created features"""
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    
    print("\n📅 TEMPORAL:")
    print(f"  - Month: {df['Month'].nunique()} unique values")
    print(f"  - Weekend games: {df['Is_Weekend'].sum()} ({df['Is_Weekend'].sum()/len(df)*100:.1f}%)")
    print(f"  - Day groups: {df['Day_Group'].value_counts().to_dict()}")
    
    print("\n🏟️ TEAM FEATURES:")
    print(f"  - Big3 home games: {df['Home_Is_Big3'].sum()}")
    print(f"  - Big3 away games: {df['Away_Is_Big3'].sum()}")
    print(f"  - Derbies: {df['Is_Derby'].sum()}")
    
    print("\n📊 HISTORICAL:")
    print(f"  - Home avg attendance range: {df['Home_Avg_Attendance'].min():.0f} - {df['Home_Avg_Attendance'].max():.0f}")
    print(f"  - Away team draw range: {df['Away_Team_Draw'].min():.0f} - {df['Away_Team_Draw'].max():.0f}")
    
    print("\n🎯 TARGET:")
    print(f"  - Attendance range: {df['Attendance'].min():,d} - {df['Attendance'].max():,d}")
    print(f"  - Attendance mean: {df['Attendance'].mean():,.0f}")

if __name__ == "__main__":
    print("="*70)
    print("FEATURE ENGINEERING - LIGA PORTUGAL")
    print("="*70)
    
    # Load data
    df = load_clean_data()
    
    # Create features
    df = create_temporal_features(df)
    df = create_team_features(df)
    df = create_historical_features(df)
    df = create_matchup_features(df)
    
    # Select final features
    df_final = select_final_features(df)
    
    # Show summary
    show_feature_summary(df_final)
    
    # Save
    output_file = 'data/liga_portugal_features.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\nFile created:")
    print(f"  - {output_file}")
    print(f"\nDataset ready for modeling: {len(df_final)} matches, {len(df_final.columns)-6} features")