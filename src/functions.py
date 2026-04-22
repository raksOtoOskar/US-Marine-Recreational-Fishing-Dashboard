import pandas as pd
import requests
import glob
from typing import Dict, Optional

# --- 1. FUNKCJE POBIERANIA ---

def fetch_ocean_temp_api(state: str, month: int) -> float:
    """Pobiera temperaturę oceanu dla danego stanu i miesiąca."""
    coords = {
        'FL': (27.7, -83.5), 'CA': (35.0, -120.0), 'TX': (27.5, -96.0),
        'NY': (40.5, -73.0), 'MA': (42.0, -70.0), 'NC': (34.0, -76.0),
        'RI': (41.5, -71.5), 'CT': (41.2, -72.7), 'NJ': (40.0, -74.0)
    }
    lat, lon = coords.get(state, (40.0, -70.0))
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        base_temp = response.json().get('current_weather', {}).get('temperature', 15.0)
        seasonal_adj = -5.0 if month in [1, 2, 11, 12] else (5.0 if month in [6, 7, 8] else 0.0)
        return round(base_temp + seasonal_adj, 2)
    except:
        return 15.0

# --- 2. CZYSTE FUNKCJE TRANSFROMACJI ---

def load_data(trip_pattern: str, catch_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wczytuje i łączy pliki NOAA."""
    pliki_trip = glob.glob(trip_pattern)
    pliki_catch = glob.glob(catch_pattern)
    
    trips = pd.concat((pd.read_csv(f, low_memory=False) for f in pliki_trip), ignore_index=True)
    catches = pd.concat((pd.read_csv(f, low_memory=False) for f in pliki_catch), ignore_index=True)

    mapowanie_trip = {'ID_CODE': 'trip_id', 'ST': 'state_fips', 'YEAR': 'year', 'MONTH': 'month', 'WAVE': 'wave'}
    mapowanie_catch = {'ID_CODE': 'trip_id', 'COMMON': 'species_name', 'CLAIM': 'catch_count', 'WGT_AB1': 'catch_weight'}

    trips = trips.rename(columns=mapowanie_trip)
    catches = catches.rename(columns=mapowanie_catch)
    
    trips['trip_id'] = trips['trip_id'].astype(str)
    catches['trip_id'] = catches['trip_id'].astype(str)
    
    # Pełne mapowanie kodów FIPS dla stanów i terytoriów nadmorskich USA
    kody_stanow = {
        1: 'AL',   # Alabama
        2: 'AK',   # Alaska
        6: 'CA',   # California
        9: 'CT',   # Connecticut
        10: 'DE',  # Delaware
        12: 'FL',  # Florida
        13: 'GA',  # Georgia
        15: 'HI',  # Hawaii (Hawaje)
        22: 'LA',  # Louisiana
        23: 'ME',  # Maine
        24: 'MD',  # Maryland
        25: 'MA',  # Massachusetts
        28: 'MS',  # Mississippi
        33: 'NH',  # New Hampshire
        34: 'NJ',  # New Jersey
        36: 'NY',  # New York
        37: 'NC',  # North Carolina
        41: 'OR',  # Oregon
        44: 'RI',  # Rhode Island
        45: 'SC',  # South Carolina
        48: 'TX',  # Texas
        51: 'VA',  # Virginia
        53: 'WA',  # Washington
        72: 'PR',  # Puerto Rico (Portoryko)
        78: 'VI'   # Virgin Islands (Wyspy Dziewicze)
    }
    
    if 'state_fips' in trips.columns:
        # Jeśli pojawi się jakiś kod spoza listy, wyświetli się jego numer, 
        # a nie słowo 'INNY', co ułatwi ewentualne debugowanie w przyszłości
        trips['state'] = trips['state_fips'].map(kody_stanow).fillna(trips['state_fips'].astype(str))
        
    catches['catch_weight'] = pd.to_numeric(catches['catch_weight'], errors='coerce').fillna(0)

    return trips[['trip_id', 'state', 'year', 'month', 'wave']], catches[['trip_id', 'species_name', 'catch_count', 'catch_weight']]

def join_noaa_tables(df_tuple: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    trips, catches = df_tuple
    return pd.merge(trips, catches, on='trip_id', how='inner')

def enrich_with_weather_api(df: pd.DataFrame) -> pd.DataFrame:
    unique_comb = df[['state', 'month']].drop_duplicates()
    unique_comb['ocean_temp_c'] = unique_comb.apply(lambda row: fetch_ocean_temp_api(row['state'], row['month']), axis=1)
    return pd.merge(df, unique_comb, on=['state', 'month'], how='left')

# --- 3. PIPELINE ---

def build_analytical_dataset(trip_pattern: str, catch_pattern: str) -> pd.DataFrame:
    """Zaktualizowany pipeline - bez scrapingu tekstu do tabeli."""
    
    # 1. Wczytujemy pliki (zwraca krotkę dwóch tabel) i od razu je łączymy
    zlaczone_dane = join_noaa_tables(load_data(trip_pattern, catch_pattern))
    
    # 2. Używamy natywnego pandasowego .pipe() do dodania API pogodowego
    final_df = zlaczone_dane.pipe(enrich_with_weather_api)
    
    return final_df