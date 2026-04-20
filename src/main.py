import streamlit as st
import pandas as pd
import os

# Importy z naszych lokalnych modułów (używamy względnych ścieżek, zakładając uruchomienie z src/ lub katalogu głównego)
from functions import build_analytical_dataset
from visualizations import (
    rysuj_mape_polowow, rysuj_trend_wyjazdow, 
    rysuj_heatmape_korelacji, rysuj_popularne_gatunki
)
from analysis import perform_state_clustering, train_catch_weight_model, predict_weight

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Dashboard Wędkarstwa Morskiego USA",
    page_icon="🎣",
    layout="wide"
)

# --- FUNKCJA WCZYTUJĄCA DANE (CACHE) ---
# Używamy st.cache_data, aby nie przetwarzać danych przy każdym kliknięciu w dashboardzie
@st.cache_data
def zaladuj_dane():
    # Definiowanie ścieżek do danych (dostosowane do struktury folderów)
    # Sprawdzamy, czy jesteśmy w src/ czy w głównym folderze
    baza_sciezki = "../data/raw" if os.path.exists("../data/raw") else "data/raw"
    
    sciezka_trip = f"{baza_sciezki}/mrip_trip_mock.csv"
    sciezka_catch = f"{baza_sciezki}/mrip_catch_mock.csv"
    
    try:
        df = build_analytical_dataset(sciezka_trip, sciezka_catch)
        return df
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return pd.DataFrame()

# --- GŁÓWNY UKŁAD DASHBOARDU ---
st.title("🎣 Analityka Wędkarstwa Morskiego w USA (NOAA MRIP)")
st.markdown("Ten panel analityczny prezentuje trendy, korelacje środowiskowe oraz predykcje oparte o uczenie maszynowe dotyczące połowów morskich w USA.")

with st.spinner("Przetwarzanie danych i pobieranie informacji z API..."):
    df_glowny = zaladuj_dane()

if df_glowny.empty:
    st.warning("Brak danych do wyświetlenia. Upewnij się, że wygenerowałeś pliki mockowe.")
    st.stop()

# Pasek boczny na filtry
st.sidebar.header("Filtry Danych")
wybrany_stan = st.sidebar.multiselect(
    "Wybierz Stan(y):",
    options=df_glowny['state'].unique(),
    default=df_glowny['state'].unique()
)

# Filtrowanie danych na podstawie wyboru użytkownika
df_filtrowany = df_glowny[df_glowny['state'].isin(wybrany_stan)]

# Zakładki dla czytelności interfejsu
zakladka_wiz, zakladka_ml, zakladka_dane = st.tabs(["📊 Przegląd Wizualny", "🤖 Analiza Zaawansowana (ML)", "📁 Surowe Dane"])

# --- ZAKŁADKA 1: WIZUALIZACJE ---
with zakladka_wiz:
    st.subheader("Geografia i Trendy Sezonowe")
    kolumna1, kolumna2 = st.columns(2)
    
    with kolumna1:
        # Plotly
        st.plotly_chart(rysuj_mape_polowow(df_filtrowany), use_container_width=True)
    
    with kolumna2:
        # Matplotlib
        st.pyplot(rysuj_trend_wyjazdow(df_filtrowany))

    st.divider()
    
    st.subheader("Korelacje Środowiskowe i Biologia")
    kolumna3, kolumna4 = st.columns(2)
    
    with kolumna3:
        # Altair
        st.altair_chart(rysuj_popularne_gatunki(df_filtrowany), use_container_width=True)
        
    with kolumna4:
        # Seaborn
        st.pyplot(rysuj_heatmape_korelacji(df_filtrowany))

# --- ZAKŁADKA 2: UCZENIE MASZYNOWE ---
with zakladka_ml:
    st.subheader("1. Klasteryzacja Stanów (K-Means)")
    st.markdown("Grupowanie stanów na podstawie aktywności wędkarskiej i średnich wag połowów.")
    df_klastry = perform_state_clustering(df_glowny)
    st.dataframe(df_klastry[['state', 'total_trips', 'avg_catch_weight', 'cluster_label']], use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Predykcja Wagi Połowu (Regresja Ridge)")
    st.markdown("Wytrenowano model przewidujący wagę na podstawie stanu, sezonu (fali) i gatunku.")
    
    # Trenowanie modelu na całym zbiorze (cache'owane w tle, tu uproszczone dla UI)
    model, wynik_r2 = train_catch_weight_model(df_glowny)
    st.info(f"Dokładność modelu (R² Score) na danych treningowych: **{wynik_r2}**")
    
    # Interaktywny formularz predykcyjny
    st.markdown("#### Sprawdź własną predykcję:")
    k1, k2, k3 = st.columns(3)
    z_stan = k1.selectbox("Stan", df_glowny['state'].unique())
    z_fala = k2.slider("Fala (Miesiące)", 1, 6, 3)
    z_gatunek = k3.selectbox("Gatunek", df_glowny['species_name'].unique())
    
    if st.button("Przewiduj Wagę"):
        predykcja = predict_weight(model, z_fala, z_stan, z_gatunek)
        st.success(f"Przewidywana waga dla pojedynczego połowu to około: **{predykcja} lbs/kg**")

# --- ZAKŁADKA 3: DANE ---
with zakladka_dane:
    st.subheader("Oczyszczony Zbiór Danych (Połączony z API i Scrapingiem)")
    st.dataframe(df_filtrowany.head(100), use_container_width=True)