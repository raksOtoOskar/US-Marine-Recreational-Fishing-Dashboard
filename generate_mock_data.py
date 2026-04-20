import pandas as pd
import numpy as np
import os

def generate_mock_noaa_data(num_trips=2000):
    np.random.seed(42)
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Generowanie Trip Data
    states = ['FL', 'CA', 'TX', 'NY', 'MA', 'NC']
    years = [2021, 2022, 2023]
    months = np.arange(1, 13)
    
    trip_data = pd.DataFrame({
        'trip_id': [f"TRP_{i:05d}" for i in range(1, num_trips + 1)],
        'state': np.random.choice(states, num_trips),
        'year': np.random.choice(years, num_trips),
        'month': np.random.choice(months, num_trips)
    })
    # Fala (Wave) to dwumiesięczny okres raportowania NOAA (np. 1 = Jan/Feb)
    trip_data['wave'] = np.ceil(trip_data['month'] / 2).astype(int)
    
    # 2. Generowanie Catch Data
    species_list = ['Red Snapper', 'Striped Bass', 'Mahi Mahi', 'Flounder', 'Tuna']
    
    # Załóżmy, że każdy wyjazd ma od 1 do 3 rekordów połowów
    catch_records = []
    for trip_id in trip_data['trip_id']:
        num_catches = np.random.randint(1, 4)
        for _ in range(num_catches):
            catch_records.append({
                'trip_id': trip_id,
                'species_name': np.random.choice(species_list),
                'catch_count': np.random.randint(1, 15),
                'catch_weight': round(np.random.uniform(0.5, 30.0), 2) # waga w kg/lbs
            })
            
    catch_data = pd.DataFrame(catch_records)
    
    # Zapis do plików
    trip_data.to_csv("data/raw/mrip_trip_mock.csv", index=False)
    catch_data.to_csv("data/raw/mrip_catch_mock.csv", index=False)
    
    print("Mock data generated successfully in 'data/raw/'")

if __name__ == "__main__":
    generate_mock_noaa_data()