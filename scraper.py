import requests
import pandas as pd
from datetime import datetime
import time
import os

# Create data folder
os.makedirs('data', exist_ok=True)

def get_fotmob_liga_portugal(season='2022-2023'):
    """Fetch Liga Portugal matches"""
    league_id = 61
    season_map = {'2022-2023': '2022/2023', '2023-2024': '2023/2024', 
                  '2024-2025': '2024/2025', '2025-2026': '2025/2026'}
    
    season_formatted = season_map[season]
    url = f'https://www.fotmob.com/api/leagues?id={league_id}&season={season_formatted.replace("/", "%2F")}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.fotmob.com/'
    }
    
    print(f"Fetching {season}...", end=' ')
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        matches = []
        if 'fixtures' in data and 'allMatches' in data['fixtures']:
            for match in data['fixtures']['allMatches']:
                utc = match.get('status', {}).get('utcTime', '')
                matches.append({
                    'ID': match.get('id'),
                    'Date_UTC': utc,
                    'Home': match.get('home', {}).get('name', ''),
                    'Away': match.get('away', {}).get('name', ''),
                    'Round': match.get('round', '')
                })
        
        if not matches:
            print("No matches")
            return None
        
        df = pd.DataFrame(matches)
        
        def parse_date(utc_str):
            if not utc_str:
                return pd.NaT
            return datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
        
        df['DateTime'] = df['Date_UTC'].apply(parse_date)
        df['Date'] = df['DateTime'].dt.strftime('%Y-%m-%d')
        df['Day_of_Week'] = df['DateTime'].dt.strftime('%A').str[:3]
        df = df.drop(columns=['DateTime'])
        
        print(f"Found {len(df)} matches")
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_match_stats(match_id):
    """Extract match stats"""
    url = f'https://www.fotmob.com/api/matchDetails?matchId={match_id}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
               'Accept': 'application/json', 'Referer': 'https://www.fotmob.com/'}
    
    stats = {'Attendance': '', 'Capacity': '', 'Pct_Full': ''}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'content' in data and 'matchFacts' in data['content']:
            info_box = data['content']['matchFacts'].get('infoBox', {})
            stats['Attendance'] = str(info_box.get('Attendance', ''))
            
            if 'Stadium' in info_box:
                stats['Capacity'] = str(info_box['Stadium'].get('capacity', ''))
            
            if stats['Attendance'] and stats['Capacity']:
                try:
                    attendance = int(stats['Attendance']) if stats['Attendance'].isdigit() else 0
                    capacity = int(stats['Capacity']) if stats['Capacity'].isdigit() else 0
                    pct = (attendance / capacity) * 100 if capacity > 0 else 0
                    stats['Pct_Full'] = f"{pct:.1f}%"
                except:
                    pass
        return stats
    except:
        return stats

def process_matches(df, season):
    """CHECK existing + ADD only new ones"""
    csv_file = 'data/liga_portugal_fotmob.csv'
    
    print(f"\nProcessing {len(df)} matches from {season}...")
    
    # LOAD EXISTING DATA
    if os.path.exists(csv_file):
        print("Loading existing data...")
        existing_data = pd.read_csv(csv_file)
        print(f"  -> {len(existing_data)} matches already exist")
    else:
        existing_data = pd.DataFrame()
    
    # CHECK WHICH ALREADY EXIST (Date + Home)
    new_matches = []
    for _, row in df.iterrows():
        if existing_data.empty:
            new_matches.append(row)
            continue
        
        exists = existing_data[
            (existing_data['Date'] == row['Date']) &
            (existing_data['Home'] == row['Home'])
        ]
        
        if exists.empty:
            new_matches.append(row)
    
    print(f"  -> {len(new_matches)} NEW matches to process")
    print("")

    if len(new_matches) == 0:
        print("No new matches to process. Skipping.")
        return existing_data
    
    # PROCESS ONLY NEW ONES
    df_new = pd.DataFrame(new_matches)
    for col in ['Attendance', 'Capacity', 'Pct_Full']:
        df_new[col] = ''
    
    for idx, row in df_new.iterrows():
        print(f"{idx+1}/{len(df_new)}: {row['Home']} vs {row['Away']}", end='')
        
        stats = get_match_stats(row['ID'])
        df_new.at[idx, 'Attendance'] = stats['Attendance']
        df_new.at[idx, 'Capacity'] = stats['Capacity']
        df_new.at[idx, 'Pct_Full'] = stats['Pct_Full']
        
        print(f" -> {stats['Pct_Full']}")
        time.sleep(1.5)

    # JOIN OLD + NEW
    cols = ['Round', 'Date', 'Day_of_Week', 'Home', 'Away', 'Attendance', 'Capacity', 'Pct_Full']
    
    df_final = pd.concat([existing_data, df_new[cols]], ignore_index=True)
    print(f"\n{len(df_new)} new matches ADDED!")
    
    # SAVE EVERYTHING
    df_final.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"TOTAL: {len(df_final)} matches saved!")
    print(f"File: {csv_file}")
    return df_final

if __name__ == "__main__":
    print("")
    print("="*60)
    print("LIGA PORTUGAL - RECENT SEASONS")
    print("="*60)
    
    # TEST MODE: Process the first N matches (TEST_MODE = False to process all)
    TEST_MODE = False
    TEST_LIMIT = 10
    
    seasons = ['2022-2023', '2023-2024', '2024-2025', '2025-2026']
    
    total_processed = 0
    
    for season in seasons:
        print(f"\nPROCESSING {season}")
        df = get_fotmob_liga_portugal(season)
        
        if df is not None:
            # Apply test limit if in test mode
            if TEST_MODE and total_processed < TEST_LIMIT:
                remaining = TEST_LIMIT - total_processed
                df = df.head(remaining)
                print(f"TEST MODE: Processing {len(df)} matches")
            
            df_final = process_matches(df, season)
            
            if TEST_MODE:
                total_processed += len(df)
                print(f"Total processed so far: {total_processed}/{TEST_LIMIT}")
                
                if total_processed >= TEST_LIMIT:
                    print(f"\nTEST LIMIT REACHED ({TEST_LIMIT} matches). Stopping.")
                    print("")
                    break
        else:
            print(f"Error in season {season}")
        
        time.sleep(2)  # Pause between seasons
    
    if not TEST_MODE:
        print(f"\nALL SEASONS COMPLETED!")