import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd

def rysuj_mape_polowow(df: pd.DataFrame):
    """
    Tworzy interaktywną mapę USA (Plotly Choropleth) pokazującą intensywność połowów.
    """
    # Agregacja liczby wyjazdów dla każdego stanu
    dane_mapy = df.groupby('state')['trip_id'].nunique().reset_index()
    dane_mapy.columns = ['Stan', 'Liczba_wyjazdow']

    fig = px.choropleth(
        dane_mapy,
        locations='Stan',
        locationmode="USA-states",
        color='Liczba_wyjazdow',
        scope="usa",
        color_continuous_scale="Blues",
        title="Intensywność wyjazdów wędkarskich wg stanów",
        labels={'Liczba_wyjazdow': 'Liczba Wyjazdów'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def rysuj_trend_wyjazdow(df: pd.DataFrame):
    """
    Tworzy wykres liniowy (Matplotlib) pokazujący trend wyjazdów w poszczególnych falach (miesiącach).
    """
    trend = df.groupby('wave')['trip_id'].nunique()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend.index, trend.values, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    
    ax.set_title('Trend wyjazdów wędkarskich na przestrzeni sezonu (Fal)', fontsize=14)
    ax.set_xlabel('Fala (Dwumiesięczny okres raportowania)', fontsize=12)
    ax.set_ylabel('Liczba unikalnych wyjazdów', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Dostosowanie osi X, aby pokazywała tylko pełne liczby fal
    ax.set_xticks(trend.index)
    
    return fig

def rysuj_heatmape_korelacji(df: pd.DataFrame):
    """
    Tworzy heatmapę (Seaborn) pokazującą korelacje między temperaturą oceanu a wynikami połowów.
    """
    # Wybieramy tylko kolumny numeryczne do korelacji
    kolumny_numeryczne = ['ocean_temp_c', 'catch_weight', 'catch_count']
    dane_korelacji = df[kolumny_numeryczne].corr()
    
    # Zmiana nazw dla lepszej czytelności na wykresie
    dane_korelacji.columns = ['Temp. Oceanu', 'Waga Połowu', 'Liczba Ryb']
    dane_korelacji.index = ['Temp. Oceanu', 'Waga Połowu', 'Liczba Ryb']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        dane_korelacji, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5,
        cbar_kws={'label': 'Współczynnik korelacji Pearsona'},
        ax=ax
    )
    ax.set_title('Korelacja: Temperatura vs Wyniki Połowów', pad=20)
    
    return fig

def rysuj_popularne_gatunki(df: pd.DataFrame):
    """
    Tworzy interaktywny wykres słupkowy (Altair) najpopularniejszych gatunków ryb.
    """
    popularnosc = df.groupby('species_name')['catch_count'].sum().reset_index()
    popularnosc = popularnosc.sort_values(by='catch_count', ascending=False).head(10)
    
    wykres = alt.Chart(popularnosc).mark_bar(color='#2ca02c').encode(
        x=alt.X('catch_count:Q', title='Łączna liczba złowionych sztuk'),
        y=alt.Y('species_name:N', sort='-x', title='Gatunek ryby'),
        tooltip=['species_name', 'catch_count']
    ).properties(
        title='Najpopularniejsze gatunki morskie',
        height=400
    ).interactive()
    
    return wykres