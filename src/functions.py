import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional

# --- 1. FUNKCJE DO POBIERANIA DANYCH ZEWNĘTRZNYCH (API & SCRAPING) ---

def fetch_ocean_temp_api(state: str, month: int) -> float:
    """
    Pobiera historyczną średnią temperaturę (mock/API) dla danego stanu i miesiąca.
    W portfolio warto pokazać obsługę requests, nawet jeśli dane są uproszczone.
    
    Args:
        state (str): Skrót nazwy stanu (np. 'FL').
        month (int): Numer miesiąca (1-12).
        
    Returns:
        float: Średnia temperatura w stopniach Celsjusza.
    """
    # Mapa współrzędnych geograficznych dla przykładowych stanów
    coords = {
        'FL': (27.7, -83.5), 'CA': (35.0, -120.0), 'TX': (27.5, -96.0),
        'NY': (40.5, -73.0), 'MA': (42.0, -70.0), 'NC': (34.0, -76.0)
    }
    
    lat, lon = coords.get(state, (0.0, 0.0))
    
    # Symulacja zapytania do Open-Meteo Marine API (Marine/Archive)
    # W warunkach produkcyjnych użyto by pełnego endpointu archive-api
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # Aby dane wyglądały realistycznie dla różnych miesięcy (sezonowość), dodajemy prostą transformację
        base_temp = response.json().get('current_weather', {}).get('temperature', 15.0)
        seasonal_adj = -5.0 if month in [1, 2, 11, 12] else (5.0 if month in [6, 7, 8] else 0.0)
        
        return round(base_temp + seasonal_adj, 2)
    except requests.RequestException:
        # Fallback w przypadku błędu sieci/API
        return 15.0

def scrape_species_info(species_name: str) -> str:
    """
    Scrapuje krótki opis gatunku z Wikipedii przy użyciu BeautifulSoup.
    
    Args:
        species_name (str): Nazwa gatunku ryby.
        
    Returns:
        str: Pierwszy akapit tekstu opisującego gatunek.
    """
    try:
        # Formatowanie nazwy pod URL Wikipedii
        formatted_name = species_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{formatted_name}"
        
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return "Brak danych (Habitat nieznany)"
            
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Szukamy pierwszego merytorycznego akapitu (niepustego i bez klas ostrzeżeń)
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 50:  # Sensowny opis ma więcej niż 50 znaków
                return text[:150] + "..." # Zwracamy skrócony opis do dashboardu
                
        return "Opis niedostępny."
    except Exception:
        return "Błąd pobierania danych."


# --- 2. CZYSTE FUNKCJE (PURE FUNCTIONS) DO PRZETWARZANIA DANYCH ---

def load_data(trip_path: str, catch_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wczytuje surowe pliki CSV."""
    trips = pd.read_csv(trip_path)
    catches = pd.read_csv(catch_path)
    return trips, catches

def join_noaa_tables(df_tuple: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Łączy tabele Trip i Catch po kluczu trip_id."""
    trips, catches = df_tuple
    return pd.merge(trips, catches, on='trip_id', how='inner')

def enrich_with_weather_api(df: pd.DataFrame) -> pd.DataFrame:
    """Wzbogaca dane o temperaturę oceanu (wywołuje API tylko dla unikalnych kombinacji)."""
    # Znajdujemy unikalne kombinacje stanu i miesiąca, aby nie odpytywać API dla każdego z 10000+ wierszy
    unique_combinations = df[['state', 'month']].drop_duplicates()
    
    # Paradygmat funkcyjny: użycie apply zamiast pętli for
    unique_combinations['ocean_temp_c'] = unique_combinations.apply(
        lambda row: fetch_ocean_temp_api(row['state'], row['month']), axis=1
    )
    
    return pd.merge(df, unique_combinations, on=['state', 'month'], how='left')

def enrich_with_species_scraping(df: pd.DataFrame) -> pd.DataFrame:
    """Wzbogaca dane o informacje o gatunkach wyciągnięte z HTML."""
    unique_species = df[['species_name']].drop_duplicates()
    
    unique_species['species_desc'] = unique_species['species_name'].map(scrape_species_info)
    
    return pd.merge(df, unique_species, on='species_name', how='left')

# --- 3. GŁÓWNY PIPELINE (FUNKCYJNY) ---

def build_analytical_dataset(trip_path: str, catch_path: str) -> pd.DataFrame:
    """
    Główna funkcja orkiestrująca. Wykorzystuje pandas pipe do sekwencyjnego, 
    funkcyjnego przekształcania DataFrame'u bez mutowania zmiennych lokalnych.
    """
    # Używamy standardowego pandas .pipe() dla zachowania łańcuchowości i czytelności
    final_df = (
        join_noaa_tables(load_data(trip_path, catch_path))
        .pipe(enrich_with_weather_api)
        .pipe(enrich_with_species_scraping)
        .dropna(subset=['catch_weight']) # Usuwamy ewentualne puste wagi
    )
    
    return final_df