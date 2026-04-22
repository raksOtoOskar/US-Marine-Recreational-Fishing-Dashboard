import streamlit as st
import pandas as pd
import os

# Importy z naszych lokalnych modułów (używamy względnych ścieżek, zakładając uruchomienie z src/ lub katalogu głównego)
from functions import build_analytical_dataset
from visualizations import (
    rysuj_mape_polowow, rysuj_trend_wyjazdow, 
    rysuj_preferencje_termiczne, rysuj_popularne_gatunki
)
from analysis import perform_state_clustering, train_catch_weight_model, predict_weight

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Dashboard Wędkarstwa Morskiego USA",
    layout="wide"
)

# --- FUNKCJA WCZYTUJĄCA DANE (CACHE) ---
# Używamy st.cache_data, aby nie przetwarzać danych przy każdym kliknięciu w dashboardzie
@st.cache_data
def zaladuj_dane():
    # Definiowanie ścieżek do danych
    baza_sciezki = "../data/raw" if os.path.exists("../data/raw") else "data/raw"
    
    # Używamy wzorców (patterns) zamiast pojedynczych plików
    wzorzec_trip = f"{baza_sciezki}/trip_2024*.csv"
    wzorzec_catch = f"{baza_sciezki}/catch_2024*.csv"
    
    try:
        df = build_analytical_dataset(wzorzec_trip, wzorzec_catch)
        return df
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return pd.DataFrame()

# --- GŁÓWNY UKŁAD DASHBOARDU ---
st.title("Analityka Rekreacyjnego Wędkarstwa Morskiego w USA (2024)")
st.markdown("Ten panel analityczny prezentuje trendy, korelacje środowiskowe oraz predykcje oparte o uczenie maszynowe dotyczące **rekreacyjnych** połowów morskich w USA w **2024 roku** (na podstawie oficjalnych danych NOAA MRIP).")
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
zakladka_wiz, zakladka_ml, zakladka_dane = st.tabs(["Przegląd Wizualny", "Analiza Zaawansowana (ML)", "Surowe Dane"])

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
        st.subheader("Przy jakiej temperaturze żerują dane ryby?")
        wykres_temp = rysuj_preferencje_termiczne(df_filtrowany)
        
        # Zmieniamy .plotly_chart na .pyplot
        st.pyplot(wykres_temp, use_container_width=True)


# --- ZAKŁADKA 2: UCZENIE MASZYNOWE ---
with zakladka_ml:
    st.subheader("1. Klasteryzacja Stanów (K-Means)")
    st.markdown("Grupowanie stanów na podstawie aktywności wędkarskiej i średnich wag połowów.")
    
    # Wywołanie funkcji i wyświetlenie tabeli (tego brakowało)
    df_klastry = perform_state_clustering(df_glowny)
    st.dataframe(df_klastry[['state', 'total_trips', 'avg_catch_weight', 'cluster_label']], use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Predykcja Wagi Połowu (Regresja Ridge)")
    model, wynik_r2 = train_catch_weight_model(df_glowny)
    st.info(f"Dokładność modelu (R² Score): **{wynik_r2}**")
    
    st.markdown("#### Sprawdź własną predykcję:")
    k1, k2, k3 = st.columns(3)
    
    z_stan = k1.selectbox("Wybierz Stan", df_glowny['state'].unique())
    z_fala = k2.slider("Wybierz Falę (Sezon)", 1, 6, 3)
    
    # Wyszukiwarka gatunków (st.selectbox w Streamlit ma wbudowane wyszukiwanie)
    unikalne_gatunki = df_glowny['species_name'].dropna().astype(str).unique()
    lista_gatunkow = sorted([gatunek for gatunek in unikalne_gatunki if gatunek != 'None'])
    
    z_gatunek = k3.selectbox("Zacznij wpisywać nazwę gatunku...", lista_gatunkow)
    
    if st.button("Oblicz przewidywaną wagę"):
        predykcja = predict_weight(model, z_fala, z_stan, z_gatunek)
        st.success(f"Przewidywana waga dla pojedynczego połowu: **{predykcja} lbs**")
        
        # --- NAPRAWA KOLIZJI NAZW DLA WIKIPEDII ---
        # Słownik mapujący dziwne nazwy z NOAA na poprawne hasła w Wikipedii
        wiki_wyjatki = {
            "DOLPHIN": "Mahi-mahi",
            # Jeśli w przyszłości znajdziesz inne błędne linki, dopisz je po przecinku, np:
            # "JAKAŚ INNA RYBA": "Poprawna_Nazwa_Wiki"
        }
        
        # Pobieramy nazwę z wyjątków, a jeśli jej tam nie ma, używamy standardowego formatowania
        nazwa_wiki = wiki_wyjatki.get(z_gatunek, z_gatunek.title().replace(' ', '_'))
        
        wiki_url = f"https://en.wikipedia.org/wiki/{nazwa_wiki}"
        st.markdown(f"[Dowiedz się więcej o gatunku: **{z_gatunek}** na Wikipedii]({wiki_url})")

# --- ZAKŁADKA 3: DANE ---
with zakladka_dane:
    st.subheader("Oczyszczony Zbiór Danych (NOAA + API Pogodowe)")
    # Wyświetlamy tabelę bez kolumny opisowej, jeśli by tam była
    kolumny_do_pokazania = [c for c in df_filtrowany.columns if c != 'species_desc']
    st.dataframe(df_filtrowany[kolumny_do_pokazania], use_container_width=True)