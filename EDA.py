import pandas as pd
import numpy as np

def load_and_clean_data():
    """Load data and remove invalid entries"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv('data/liga_portugal_fotmob.csv')
    print(f"Total matches: {len(df)}")
    
    # Check missing attendance
    missing = (df['Attendance'] == '') | (df['Attendance'].isna())
    print(f"Missing attendance: {missing.sum()} ({missing.sum()/len(df)*100:.1f}%)")
    
    # Remove missing
    df_clean = df[~missing].copy()
    print(f"Valid matches: {len(df_clean)}")
    
    # Convert to numeric
    df_clean['Attendance'] = pd.to_numeric(df_clean['Attendance'], errors='coerce')
    
    # Remove if conversion failed
    failed = df_clean['Attendance'].isna()
    if failed.sum() > 0:
        print(f"Failed to convert: {failed.sum()}")
        df_clean = df_clean[~failed].copy()
    
    # Remove outliers (attendance = 0)
    zeros = df_clean['Attendance'] == 0
    if zeros.sum() > 0:
        print(f"Zero attendance: {zeros.sum()}")
        df_clean = df_clean[~zeros].copy()
    
    # Convert to int
    df_clean['Attendance'] = df_clean['Attendance'].astype(int)
    
    print(f"\nFinal clean dataset: {len(df_clean)} matches")
    
    # Save clean version (without Capacity and Pct_Full)
    cols_to_save = ['Round', 'Date', 'Day_of_Week', 'Home', 'Away', 'Attendance']
    df_clean[cols_to_save].to_csv('data/liga_portugal_clean.csv', index=False, encoding='utf-8')
    print("Saved: data/liga_portugal_clean.csv")
    
    return df_clean

def basic_stats(df):
    """Basic statistics"""
    print("\n" + "="*70)
    print("ATTENDANCE STATISTICS")
    print("="*70)
    
    print(f"\nMean:     {df['Attendance'].mean():>10,.0f}")
    print(f"Median:   {df['Attendance'].median():>10,.0f}")
    print(f"Std Dev:  {df['Attendance'].std():>10,.0f}")
    print(f"Min:      {df['Attendance'].min():>10,d}")
    print(f"Max:      {df['Attendance'].max():>10,d}")
    
    print(f"\n25th percentile: {df['Attendance'].quantile(0.25):>10,.0f}")
    print(f"75th percentile: {df['Attendance'].quantile(0.75):>10,.0f}")

def analyze_by_team(df):
    """Analyze by home team"""
    print("\n" + "="*70)
    print("TOP 10 TEAMS - AVERAGE ATTENDANCE (HOME)")
    print("="*70)
    
    team_stats = df.groupby('Home').agg({
        'Attendance': ['mean', 'median', 'count']
    }).round(0)
    
    team_stats.columns = ['Avg', 'Median', 'Matches']
    team_stats = team_stats.sort_values('Avg', ascending=False)
    
    # Convert to int
    for col in ['Avg', 'Median', 'Matches']:
        team_stats[col] = team_stats[col].astype(int)
    
    print(team_stats.head(10).to_string())
    
    print("\n" + "="*70)
    print("BOTTOM 5 TEAMS - AVERAGE ATTENDANCE (HOME)")
    print("="*70)
    print(team_stats.tail(5).to_string())
    
    return team_stats

def analyze_by_day(df):
    """Analyze by day of week"""
    print("\n" + "="*70)
    print("ATTENDANCE BY DAY OF WEEK")
    print("="*70)
    
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_stats = df.groupby('Day_of_Week')['Attendance'].agg(['mean', 'median', 'count']).round(0)
    day_stats = day_stats.reindex(day_order)
    
    for day_en, day_pt in zip(day_order, day_names):
        if day_en in day_stats.index:
            mean_val = int(day_stats.loc[day_en, 'mean'])
            median_val = int(day_stats.loc[day_en, 'median'])
            count_val = int(day_stats.loc[day_en, 'count'])
            print(f"{day_pt:10s}: Avg {mean_val:>8,d} | Median {median_val:>8,d} | ({count_val:>3d} matches)")
    
    return day_stats

def analyze_by_round(df):
    """Analyze by round/jornada"""
    print("\n" + "="*70)
    print("ATTENDANCE BY ROUND (Jornada)")
    print("="*70)
    
    # Convert Round to numeric
    df['Round_Num'] = pd.to_numeric(df['Round'], errors='coerce')
    
    round_stats = df.groupby('Round_Num')['Attendance'].agg(['mean', 'count']).round(0)
    round_stats = round_stats.sort_index()
    
    # Convert to int
    round_stats['mean'] = round_stats['mean'].astype(int)
    round_stats['count'] = round_stats['count'].astype(int)
    
    print("\nFirst 10 rounds:")
    print(round_stats.head(10).to_string())
    
    print("\nLast 10 rounds:")
    print(round_stats.tail(10).to_string())
    
    # Compare first half vs second half
    mid_round = int(round_stats.index.max() // 2)
    first_half = int(df[df['Round_Num'] <= mid_round]['Attendance'].mean())
    second_half = int(df[df['Round_Num'] > mid_round]['Attendance'].mean())
    
    print(f"\nFirst half of season (rounds 1-{mid_round}): {first_half:,d}")
    print(f"Second half of season (rounds {mid_round+1}-{int(round_stats.index.max())}): {second_half:,d}")
    print(f"Difference: {second_half - first_half:+,d} ({((second_half/first_half - 1)*100):+.1f}%)")

def find_rivalries(df):
    """Find high-attendance matchups (derbies)"""
    print("\n" + "="*70)
    print("TOP 10 MATCHUPS - HIGHEST AVERAGE ATTENDANCE")
    print("="*70)
    
    df['Matchup'] = df['Home'] + ' vs ' + df['Away']
    
    matchup_stats = df.groupby('Matchup')['Attendance'].agg(['mean', 'count']).round(0)
    matchup_stats = matchup_stats[matchup_stats['count'] >= 2]  # At least 2 matches
    matchup_stats = matchup_stats.sort_values('mean', ascending=False)
    
    # Convert to int
    matchup_stats['mean'] = matchup_stats['mean'].astype(int)
    matchup_stats['count'] = matchup_stats['count'].astype(int)
    
    print(matchup_stats.head(10).to_string())

def data_coverage(df):
    """Show data coverage by season"""
    print("\n" + "="*70)
    print("DATA COVERAGE BY SEASON")
    print("="*70)
    
    # Extract year from date
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    
    # Determine season
    def get_season(date_str):
        date = pd.to_datetime(date_str)
        year = date.year
        month = date.month
        if month >= 8:  # Aug-Dec = start of season
            return f"{year}-{year+1}"
        else:  # Jan-May = end of season
            return f"{year-1}-{year}"
    
    df['Season'] = df['Date'].apply(get_season)
    
    season_stats = df.groupby('Season')['Attendance'].agg(['count', 'mean']).round(0)
    season_stats = season_stats.sort_index()
    season_stats['mean'] = season_stats['mean'].astype(int)
    season_stats.columns = ['Matches', 'Avg_Attendance']
    
    print(season_stats.to_string())

if __name__ == "__main__":
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS - LIGA PORTUGAL")
    print("="*70)
    
    # Load and clean
    df = load_and_clean_data()
    
    # Basic stats
    basic_stats(df)
    
    # By team
    team_stats = analyze_by_team(df)
    
    # By day
    day_stats = analyze_by_day(df)
    
    # By round
    analyze_by_round(df)
    
    # Rivalries
    find_rivalries(df)
    
    # Coverage
    data_coverage(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nFile created:")
    print("  - data/liga_portugal_clean.csv  (clean dataset for modeling)")
    print(f"\nDataset ready: {len(df)} matches with valid attendance data")