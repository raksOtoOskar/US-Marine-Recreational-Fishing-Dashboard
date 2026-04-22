# US Marine Recreational Fishing Analytics (2024)

Interaktywny panel analityczny (Dashboard) zbudowany w Pythonie (Streamlit), służący do eksploracji, wizualizacji i analizy predykcyjnej danych o rekreacyjnych połowach morskich w USA. Projekt bazuje na oficjalnych danych federalnych NOAA MRIP.

## Główne funkcjonalności

* **Rozbudowany proces ETL:** Łączenie wielkoskalowych tabel podróży i połowów, zaawansowane mapowanie terytorialnych kodów FIPS (w tym Hawaje i Portoryko).
* **Integracja z zewnętrznymi API:**
  * **Open-Meteo API:** Dynamiczne pobieranie danych o temperaturach wód oceanicznych (SST) dla wskazanych współrzędnych.
  * **Wikipedia REST API:** Automatyczne odnośniki do gatunków ryb z systemem obsługi kolizji nazewnictwa systematycznego (np. Dolphin -> Mahi-mahi).
* **Uczenie Maszynowe (Scikit-Learn):**
  * **Klasteryzacja (K-Means):** Segmentacja stanów na 4 grupy aktywności wędkarskiej (z poprawną izolacją wartości odstających/outlierów).
  * **Regresja Ridge:** Interaktywny predyktor wagi pojedynczego połowu bazujący na sezonie, stanie i wybranym gatunku (R^2 ~ 0.30).
* **Zaawansowana Wizualizacja:** Wykorzystanie 4 różnych bibliotek (Plotly, Seaborn, Matplotlib, Altair) do analizy przestrzennej (mapy Choropleth), czasowej oraz wyznaczania okien biotopowych (wykresy pudełkowe preferencji termicznych).

## Technologie

* **Język:** Python 3.9+
* **Przetwarzanie danych:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Wizualizacja:** Plotly, Seaborn, Matplotlib, Altair
* **Interfejs & Wdrożenie:** Streamlit
* **Pobieranie danych:** Requests (REST API)

## Jak uruchomić projekt lokalnie?

1. Sklonuj repozytorium:
git clone https://github.com/TwojaNazwaUzytkownika/US-Marine-Fishing-Dashboard.git
cd US-Marine-Fishing-Dashboard

2. Zainstaluj wymagane biblioteki:
pip install -r requirements.txt

3. Uruchom aplikację:
streamlit run src/main.py
