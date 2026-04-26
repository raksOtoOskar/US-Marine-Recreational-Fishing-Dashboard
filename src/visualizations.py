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

def rysuj_preferencje_termiczne(df: pd.DataFrame):
    """
    Pokazuje rozkład temperatur dla top 10 gatunków.
    """
    # 1. Filtrujemy dane
    df_ryby = df[(df['species_name'].notna()) & (df['species_name'].astype(str) != 'None')].copy()
    top_gatunki = df_ryby['species_name'].value_counts().nlargest(10).index
    df_top = df_ryby[df_ryby['species_name'].isin(top_gatunki)]
    
    # 2. Tworzymy obiekt wykresu 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. Rysujemy wykres
    sns.boxplot(
        data=df_top,
        x='ocean_temp_c',
        y='species_name',
        palette='crest',
        ax=ax
    )
    
    # 4. Kosmetyka wykresu
    ax.set_title('Preferencje termiczne (Okna temperaturowe) dla top 10 gatunków', fontsize=14, pad=15)
    ax.set_xlabel('Temperatura Oceanu (°C)', fontsize=12)
    ax.set_ylabel('Gatunek Ryby', fontsize=12)
    
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Zwracamy gotową figurę
    return fig

def rysuj_popularne_gatunki(df: pd.DataFrame):
    """
    Tworzy interaktywny wykres słupkowy (Altair) najpopularniejszych gatunków ryb.
    """
    # Na wykresie pokazujemy tylko prawdziwe ryby, odrzucamy puste przeloty
    df_ryby = df[(df['species_name'].notna()) & (df['species_name'].astype(str) != 'None')]
    
    popularnosc = df_ryby.groupby('species_name')['catch_count'].sum().reset_index()
    popularnosc = popularnosc.sort_values(by='catch_count', ascending=False).head(10)
    
    popularnosc['catch_count'] = popularnosc['catch_count'].round().astype(int)
    
    wykres = alt.Chart(popularnosc).mark_bar(color='#2ca02c').encode(
        x=alt.X('catch_count:Q', title='Łączna liczba złowionych sztuk'),
        y=alt.Y('species_name:N', sort='-x', title='Gatunek ryby'),
        tooltip=['species_name', 'catch_count']
    ).properties(
        title='Najpopularniejsze gatunki morskie',
        height=400
    ).interactive()
    
    return wykres