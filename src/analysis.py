import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any

def perform_state_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Dokonuje segmentacji (K-Means) stanów nadmorskich na podstawie:
    - Całkowitej liczby wyjazdów
    - Średniej wagi połowu
    """
    # Agregacja danych do poziomu stanu
    state_metrics = df.groupby('state').agg(
        total_trips=('trip_id', 'nunique'),
        avg_catch_weight=('catch_weight', 'mean'),
        total_catch_count=('catch_count', 'sum')
    ).reset_index()
    
    # Wyciągamy cechy do uczenia
    features = state_metrics[['total_trips', 'avg_catch_weight', 'total_catch_count']]
    
    # Skalowanie danych
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Algorytm K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    state_metrics['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Dynamiczne przypisywanie etykiet - teraz dla 4 klastrów
    srednie_klastrow = state_metrics.groupby('cluster')['total_trips'].mean().sort_values()
    posortowane_id = srednie_klastrow.index.tolist()
    
    # Dodajemy nową gradację
    cluster_names = {
        posortowane_id[0]: "Bardzo Niska Aktywność",
        posortowane_id[1]: "Niska Aktywność",
        posortowane_id[2]: "Umiarkowana Aktywność",
        posortowane_id[3]: "Wysoka Aktywność"
    }
    
    state_metrics['cluster_label'] = state_metrics['cluster'].map(cluster_names).fillna("Inne")
    
    return state_metrics

def train_catch_weight_model(df: pd.DataFrame) -> tuple[Pipeline, float]:
    """
    Trenuje model regresji grzbietowej (Ridge Regression) przewidujący
    wagę pojedynczego połowu na podstawie sezonu (wave), stanu i gatunku.
    
    Args:
        df (pd.DataFrame): Pełny, połączony zbiór analityczny.
        
    Returns:
        tuple: Wytrenowany pipeline (model) oraz wynik R^2 (dokładność) na zbiorze treningowym.
    """
    # Model uczy się tylko na udanych połowach (gdzie waga jest większa niż 0 i znany jest gatunek)
    df_udane = df[(df['catch_weight'] > 0) & (df['species_name'].notna()) & (df['species_name'] != 'None')].copy()

    # Wybór cech (Features) na podstawie PRZEFILTROWANEGO zbioru
    X = df_udane[['wave', 'state', 'species_name']]
    y = df_udane['catch_weight']
    
    # Tworzymy transformator dla zmiennych kategorycznych (stan, gatunek)
    categorical_features = ['state', 'species_name']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Pozostaw zmienną 'wave' bez zmian (liczbową)
    )
    
    # Budowa potoku (Pipeline) zapobiegającego wyciekom danych (data leakage)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Trening modelu
    model.fit(X, y)
    
    # Ocena modelu (R^2 Score)
    score = model.score(X, y)
    
    return model, round(score, 3)

def predict_weight(model: Pipeline, wave: int, state: str, species: str) -> float:
    """
    Czysta funkcja wnioskująca (Inference). 
    Zwraca przewidywaną wagę dla podanych parametrów wyjazdu.
    """
    input_data = pd.DataFrame({
        'wave': [wave],
        'state': [state],
        'species_name': [species]
    })
    
    prediction = model.predict(input_data)[0]
    return round(max(0.1, prediction), 2) # Zabezpieczenie przed ujemną wagą