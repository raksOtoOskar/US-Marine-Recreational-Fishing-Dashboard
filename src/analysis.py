import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any

def perform_state_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Dokonuje segmentacji (K-Means) stanów nadmorskich na podstawie:
    - Całkowitej liczby wyjazdów
    - Średniej wagi połowu
    
    Args:
        df (pd.DataFrame): Przetworzony zbiór danych NOAA.
        n_clusters (int): Liczba klastrów.
        
    Returns:
        pd.DataFrame: Zbiór stanów z przypisanymi etykietami klastrów.
    """
    # Agregacja danych do poziomu stanu
    state_metrics = df.groupby('state').agg(
        total_trips=('trip_id', 'nunique'),
        avg_catch_weight=('catch_weight', 'mean'),
        total_catch_count=('catch_count', 'sum')
    ).reset_index()
    
    # Wyciągamy cechy do uczenia
    features = state_metrics[['total_trips', 'avg_catch_weight', 'total_catch_count']]
    
    # Skalowanie danych (wymagane w K-Means)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Algorytm K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    state_metrics['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Zamiana id klastra na czytelną etykietę dla dashboardu
    cluster_names = {0: "Wysoka Aktywność", 1: "Umiarkowane Połowy", 2: "Niska Aktywność"}
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
    # Wybór cech (Features) i zmiennej objaśnianej (Target)
    X = df[['wave', 'state', 'species_name']]
    y = df['catch_weight']
    
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