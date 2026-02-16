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
    
    # Sort by date (important for historical features)
    df = df.sort_values('Date_parsed').reset_index(drop=True)
    
    # Day type (weekend vs weekday vs rare)
    df['Day_Type'] = df['Day_of_Week'].map({
        'Fri': 'weekend',
        'Sat': 'weekend',
        'Sun': 'weekend',
        'Mon': 'weekday',
        'Tue': 'rare',
        'Wed': 'rare',
        'Thu': 'rare'
    })
    
    # Round number
    df['Round_Num'] = pd.to_numeric(df['Round'], errors='coerce').astype(int)
    
    print("  ✓ Temporal features created")
    
    return df

def create_team_features(df):
    """Create team-based features"""
    print("\nCreating team features...")
    
    # Big 3 teams
    big_3 = ['Benfica', 'FC Porto', 'Sporting CP']
    df['Home_Is_Big3'] = df['Home'].isin(big_3).astype(int)
    df['Away_Is_Big3'] = df['Away'].isin(big_3).astype(int)
    
    print("  ✓ Team features created")
    
    return df

def create_historical_features(df):
    """Create historical average features (NO DATA LEAKAGE)"""
    print("\nCreating historical features...")
    
    # Home team historical average (expanding mean - only PAST games)
    df['Home_Avg_Attendance'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.expanding().mean().shift(1)  # shift(1) excludes current game
    )
    
    # Recent form (last 3 home games average)
    df['Home_Last3_Avg'] = df.groupby('Home')['Attendance'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    
    # Fill NaN (first games of each team) with overall mean
    overall_mean = df['Attendance'].mean()
    df['Home_Avg_Attendance'] = df['Home_Avg_Attendance'].fillna(overall_mean)
    df['Home_Last3_Avg'] = df['Home_Last3_Avg'].fillna(overall_mean)
    
    # Convert to int (no need for decimal precision)
    df['Home_Avg_Attendance'] = df['Home_Avg_Attendance'].round().astype(int)
    df['Home_Last3_Avg'] = df['Home_Last3_Avg'].round().astype(int)
    
    print("  ✓ Historical features created")
    
    return df

def create_matchup_features(df):
    """Create matchup-specific features (captures rivalries)"""
    print("\nCreating matchup features...")
    
    # Create matchup identifier
    df['Matchup'] = df['Home'] + ' vs ' + df['Away']
    
    # Historical average for THIS specific matchup (WITHOUT DATA LEAKAGE)
    df['Matchup_Avg_Attendance'] = df.groupby('Matchup')['Attendance'].transform(
        lambda x: x.expanding().mean().shift(1)  # Only past games of this matchup
    )
    
    # Fill NaN with Home_Avg_Attendance (fallback for first matchup)
    df['Matchup_Avg_Attendance'] = df['Matchup_Avg_Attendance'].fillna(df['Home_Avg_Attendance'])
    
    # Convert to int
    df['Matchup_Avg_Attendance'] = df['Matchup_Avg_Attendance'].round().astype(int)
    
    print("  ✓ Matchup features created (captures rivalries)")
    
    return df

def select_final_features(df):
    """Select final features for modeling"""
    print("\nSelecting final features...")
    
    feature_cols = [
        # Target 
        'Attendance',
        
        # Identifiers
        'Date',
        'Home',
        'Away',
        
        # Features for modeling 
        'Home_Avg_Attendance',      
        'Home_Last3_Avg',           
        'Matchup_Avg_Attendance',   
        'Home_Is_Big3',             
        'Away_Is_Big3',             
        'Day_Type',                 
        'Round_Num'                 
    ]
    
    df_final = df[feature_cols].copy()
    
    print(f"  ✓ {len(feature_cols)} columns selected")
    
    return df_final

def show_summary(df):
    """Show summary of features"""
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    
    print(f"\nDataset: {len(df)} matches")
    print(f"Features: 7 (for modeling)")
    
    print(f"\nFeature breakdown:")
    print(f"  • Historical: 3 (Home_Avg, Home_Last3, Matchup_Avg)")
    print(f"  • Team: 2 (Home_Is_Big3, Away_Is_Big3)")
    print(f"  • Temporal: 2 (Day_Type, Round_Num)")
    
    print(f"\nDay distribution:")
    day_counts = df['Day_Type'].value_counts()
    for day_type, count in day_counts.items():
        print(f"  • {day_type}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nBig3 teams:")
    print(f"  • Home Big3: {df['Home_Is_Big3'].sum()} matches")
    print(f"  • Away Big3: {df['Away_Is_Big3'].sum()} matches")
    print(f"  • Big games (both Big3): {((df['Home_Is_Big3']==1) & (df['Away_Is_Big3']==1)).sum()}")
    
    print(f"\nHistorical features (int):")
    print(f"  • Home avg range: {df['Home_Avg_Attendance'].min():,d} - {df['Home_Avg_Attendance'].max():,d}")
    print(f"  • Last 3 avg range: {df['Home_Last3_Avg'].min():,d} - {df['Home_Last3_Avg'].max():,d}")
    print(f"  • Matchup avg range: {df['Matchup_Avg_Attendance'].min():,d} - {df['Matchup_Avg_Attendance'].max():,d}")
    
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
    
    # Select final
    df_final = select_final_features(df)
    
    # Show summary
    show_summary(df_final)
    
    # Save
    output_file = 'data/liga_portugal_features.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✓ Saved: {output_file}")
