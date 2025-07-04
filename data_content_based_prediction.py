import pandas as pd
import numpy as np

# --- 1. Dataset des notes des utilisateurs ---

data = {
    'Film A': [5, 4, np.nan, 1, 3, 4, 5, 4, 3, 2, 5, 4, 3, 4, 5, 2, 1, np.nan, 4, 3, 5, np.nan, 2, 4, 3],
    'Film B': [3, np.nan, 4, 2, 5, 4, 3, 4, 5, 2, np.nan, 4, 3, 5, np.nan, 4, 2, 3, 4, np.nan, 5, 3, 4, np.nan, 4],
    'Film C': [4, 5, 2, np.nan, 4, 4, 5, 3, 4, 2, 5, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 4, np.nan, 3],
    'Film D': [np.nan, 3, 5, 4, 2, np.nan, 4, 5, 3, 2, 4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 4, np.nan, 2],
    'Film E': [2, 4, np.nan, 3, np.nan, 3, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, np.nan, 3, 4, np.nan, 5, 2, 4, np.nan],
    'Film F': [np.nan, 2, np.nan, 5, 4, np.nan, 3, 4, 5, np.nan, 2, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film G': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film H': [np.nan, 5, np.nan, 3, np.nan, 2, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 3],
    'Film I': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film J': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film K': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film L': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film M': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film N': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film O': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film P': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film Q': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film R': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film S': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film T': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film U': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film V': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film W': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film X': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film Y': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film Z': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AA': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film AB': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AC': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film AD': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AE': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AF': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AG': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AH': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film AI': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AJ': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film AK': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AL': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AM': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AN': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AO': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film AP': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AQ': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film AR': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AS': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AT': [5, np.nan, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AU': [np.nan, 4, 3, 5, 2, np.nan, 4, 3, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AV': [2, 3, np.nan, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, np.nan, 5, 4, 2, np.nan, 3, 4, np.nan, 5, 2, np.nan, 4],
    'Film AW': [np.nan, 5, 4, 3, 2, 4, 3, np.nan, 5, 4, np.nan, 2, 3, 4, np.nan, 5, 2, np.nan, 4, 3, np.nan, 5, 4, np.nan, 2],
    'Film AX': [4, np.nan, 3, np.nan, 5, 4, 4, 5, np.nan, 3, 4, 2, 5, np.nan, 4, 3, 5, np.nan, 2, 4, np.nan, 3, 5, np.nan, 4],
    'Film AY': [3, 4, 5, np.nan, 2, 3, 4, np.nan, 5, 4, 3, np.nan, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3],
    'Film AZ': [4, 3, np.nan, 5, 4, 2, 3, np.nan, 4, 5, np.nan, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 2, np.nan, 5, 4, np.nan, 3]
}

index = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Marcel', 'Nina', 'Oscar', 'Paul', 'Quentin', 'Rita', 'Sophie', 'Tina', 'Ursula', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zoe', 'Aaron', 'Bella', 'Cleo', 'Diana', 'Ethan', 'Fiona']
df = pd.DataFrame(data, index=index)

print("--- Dataset des notes des utilisateurs (df) ---")
print(df.head()) # Affiche les premières lignes pour ne pas surcharger la console
print(f"\nDimensions du dataset de notes : {df.shape}")

# --- 2a. Dataset des attributs des films ---

film_attributes_data = {
    'Film A': ['Action', 'Sci-Fi'],
    'Film B': ['Comedy', 'Romance'],
    'Film C': ['Action', 'Thriller'],
    'Film D': ['Drama', 'Mystery'],
    'Film E': ['Sci-Fi', 'Adventure'],
    'Film F': ['Comedy', 'Family'],
    'Film G': ['Action', 'Adventure'],
    'Film H': ['Drama', 'Romance'],
    'Film I': ['Thriller', 'Mystery'],
    'Film J': ['Sci-Fi', 'Drama'],
    'Film K': ['Action', 'Comedy'],
    'Film L': ['Romance', 'Family'],
    'Film M': ['Mystery', 'Adventure'],
    'Film N': ['Sci-Fi', 'Thriller'],
    'Film O': ['Action', 'Drama'],
    'Film P': ['Comedy', 'Mystery'],
    'Film Q': ['Thriller', 'Adventure'],
    'Film R': ['Sci-Fi', 'Romance'],
    'Film S': ['Drama', 'Family'],
    'Film T': ['Action', 'Romance'],
    'Film U': ['Comedy', 'Thriller'],
    'Film V': ['Drama', 'Adventure'],
    'Film W': ['Sci-Fi', 'Family'],
    'Film X': ['Action', 'Mystery'],
    'Film Y': ['Romance', 'Thriller'],
    'Film Z': ['Comedy', 'Sci-Fi'],
    'Film AA': ['Adventure', 'Drama'],
    'Film AB': ['Family', 'Mystery'],
    'Film AC': ['Thriller', 'Action'],
    'Film AD': ['Romance', 'Adventure'],
    'Film AE': ['Sci-Fi', 'Mystery'],
    'Film AF': ['Comedy', 'Drama'],
    'Film AG': ['Action', 'Family'],
    'Film AH': ['Drama', 'Thriller'],
    'Film AI': ['Sci-Fi', 'Comedy'],
    'Film AJ': ['Mystery', 'Romance'],
    'Film AK': ['Adventure', 'Action'],
    'Film AL': ['Family', 'Thriller'],
    'Film AM': ['Comedy', 'Adventure'],
    'Film AN': ['Drama', 'Sci-Fi'],
    'Film AO': ['Thriller', 'Romance'],
    'Film AP': ['Action', 'Mystery'],
    'Film AQ': ['Sci-Fi', 'Family'],
    'Film AR': ['Comedy', 'Adventure'],
    'Film AS': ['Drama', 'Thriller'],
    'Film AT': ['Mystery', 'Romance'],
    'Film AU': ['Adventure', 'Sci-Fi'],
    'Film AV': ['Family', 'Comedy'],
    'Film AW': ['Thriller', 'Drama'],
    'Film AX': ['Romance', 'Action'],
    'Film AY': ['Sci-Fi', 'Mystery'],
    'Film AZ': ['Comedy', 'Thriller']
}

all_genres = sorted(list(set(genre for genres_list in film_attributes_data.values() for genre in genres_list)))

df_film_attributes = pd.DataFrame(0, index=film_attributes_data.keys(), columns=all_genres)

for film, genres_list in film_attributes_data.items():
    for genre in genres_list:
        df_film_attributes.loc[film, genre] = 1

print("\n--- DataFrame des Attributs de Films (df_film_attributes) ---")
print(df_film_attributes.head())
print(f"\nDimensions du DataFrame des Attributs : {df_film_attributes.shape}")

# --- 2b. Dataset des attributs des films plus complet ---

film_attributes_data2 = {
        'Film A': ['Action', 'Sci-Fi', 'Europe', 'Alice Smith', 'John Doe'],
        'Film B': ['Comedy', 'Romance', 'North America', 'Bob Lee', 'Jane Miller'],
        'Film C': ['Action', 'Thriller', 'Asia', 'Charlie Kim', 'Sophie Turner'],
        'Film D': ['Drama', 'Mystery', 'Europe', 'David Park', 'Lucas Brown'],
        'Film E': ['Sci-Fi', 'Adventure', 'Oceania', 'Eve White', 'Anna Scott'],
        'Film F': ['Comedy', 'Family', 'Africa', 'Marcel Dubois', 'Paul Green'],
        'Film G': ['Action', 'Adventure', 'South America', 'Nina Lopez', 'Oscar Wilde'],
        'Film H': ['Drama', 'Romance', 'Europe', 'Oscar Wilde', 'Marie Curie'],
        'Film I': ['Thriller', 'Mystery', 'Asia', 'Paul Green', 'Tina Fey'],
        'Film J': ['Sci-Fi', 'Drama', 'North America', 'Quentin Tarantino', 'Rita Ora'],
        'Film K': ['Action', 'Comedy', 'Europe', 'Sophie Turner', 'Victor Hugo'],
        'Film L': ['Romance', 'Family', 'Africa', 'Tina Fey', 'Wendy Wu'],
        'Film M': ['Mystery', 'Adventure', 'Oceania', 'Ursula Andress', 'Xavier Dolan'],
        'Film N': ['Sci-Fi', 'Thriller', 'Asia', 'Victor Hugo', 'Yara Shahidi'],
        'Film O': ['Action', 'Drama', 'Europe', 'Wendy Wu', 'Zoe Saldana'],
        'Film P': ['Comedy', 'Mystery', 'North America', 'Aaron Paul', 'Bella Hadid'],
        'Film Q': ['Thriller', 'Adventure', 'South America', 'Cleo de Merode', 'Diana Ross'],
        'Film R': ['Sci-Fi', 'Romance', 'Europe', 'Ethan Hawke', 'Fiona Apple'],
        'Film S': ['Drama', 'Family', 'Africa', 'Gina Torres', 'Hugo Weaving'],
        'Film T': ['Action', 'Romance', 'Oceania', 'Ivy Chen', 'Jack Black'],
        'Film U': ['Comedy', 'Thriller', 'Asia', 'Kurt Russell', 'Lily Collins'],
        'Film V': ['Drama', 'Adventure', 'Europe', 'Marcel Dubois', 'Nina Lopez'],
        'Film W': ['Sci-Fi', 'Family', 'North America', 'Oscar Wilde', 'Paul Green'],
        'Film X': ['Action', 'Mystery', 'Africa', 'Quentin Tarantino', 'Rita Ora'],
        'Film Y': ['Romance', 'Thriller', 'Europe', 'Sophie Turner', 'Tina Fey'],
        'Film Z': ['Comedy', 'Sci-Fi', 'Asia', 'Ursula Andress', 'Victor Hugo'],
        'Film AA': ['Adventure', 'Drama', 'Oceania', 'Wendy Wu', 'Xavier Dolan'],
        'Film AB': ['Family', 'Mystery', 'Europe', 'Yara Shahidi', 'Zoe Saldana'],
        'Film AC': ['Thriller', 'Action', 'North America', 'Aaron Paul', 'Bella Hadid'],
        'Film AD': ['Romance', 'Adventure', 'South America', 'Cleo de Merode', 'Diana Ross'],
        'Film AE': ['Sci-Fi', 'Mystery', 'Europe', 'Ethan Hawke', 'Fiona Apple'],
        'Film AF': ['Comedy', 'Drama', 'Africa', 'Gina Torres', 'Hugo Weaving'],
        'Film AG': ['Action', 'Family', 'Oceania', 'Ivy Chen', 'Jack Black'],
        'Film AH': ['Drama', 'Thriller', 'Asia', 'Kurt Russell', 'Lily Collins'],
        'Film AI': ['Sci-Fi', 'Comedy', 'Europe', 'Marcel Dubois', 'Nina Lopez'],
        'Film AJ': ['Mystery', 'Romance', 'North America', 'Oscar Wilde', 'Paul Green'],
        'Film AK': ['Adventure', 'Action', 'Africa', 'Quentin Tarantino', 'Rita Ora'],
        'Film AL': ['Family', 'Thriller', 'Europe', 'Sophie Turner', 'Tina Fey'],
        'Film AM': ['Comedy', 'Adventure', 'Asia', 'Ursula Andress', 'Victor Hugo'],
        'Film AN': ['Drama', 'Sci-Fi', 'Oceania', 'Wendy Wu', 'Xavier Dolan'],
        'Film AO': ['Thriller', 'Romance', 'Europe', 'Yara Shahidi', 'Zoe Saldana'],
        'Film AP': ['Action', 'Mystery', 'North America', 'Aaron Paul', 'Bella Hadid'],
        'Film AQ': ['Sci-Fi', 'Family', 'South America', 'Cleo de Merode', 'Diana Ross'],
        'Film AR': ['Comedy', 'Adventure', 'Europe', 'Ethan Hawke', 'Fiona Apple'],
        'Film AS': ['Drama', 'Thriller', 'Africa', 'Gina Torres', 'Hugo Weaving'],
        'Film AT': ['Mystery', 'Romance', 'Oceania', 'Ivy Chen', 'Jack Black'],
        'Film AU': ['Adventure', 'Sci-Fi', 'Asia', 'Kurt Russell', 'Lily Collins'],
        'Film AV': ['Family', 'Comedy', 'Europe', 'Marcel Dubois', 'Nina Lopez'],
        'Film AW': ['Thriller', 'Drama', 'North America', 'Oscar Wilde', 'Paul Green'],
        'Film AX': ['Romance', 'Action', 'Africa', 'Quentin Tarantino', 'Rita Ora'],
        'Film AY': ['Sci-Fi', 'Mystery', 'Europe', 'Sophie Turner', 'Tina Fey'],
        'Film AZ': ['Comedy', 'Thriller', 'Asia', 'Ursula Andress', 'Victor Hugo']
    }

all_genres2 = sorted(list(set(genre for genres_list in film_attributes_data2.values() for genre in genres_list)))

df_film_attributes2 = pd.DataFrame(0, index=film_attributes_data2.keys(), columns=all_genres2)

for film, genres_list in film_attributes_data2.items():
    for genre in genres_list:
        df_film_attributes2.loc[film, genre] = 1

print("\n--- DataFrame des Attributs de Films (df_film_attributes2) ---")
print(df_film_attributes2.head())
print(f"\nDimensions du DataFrame des Attributs : {df_film_attributes2.shape}")

# --- 2c. Dataset des attributs des films plus complet organisé ---

film_attributes_data_categorized = {
    'Film A': {'genres': ['Action', 'Sci-Fi'], 'regions': ['Europe'], 'actors': ['Alice Smith', 'John Doe'], 'directors': ['Director A']},
    'Film B': {'genres': ['Comedy', 'Romance'], 'regions': ['North America'], 'actors': ['Bob Lee', 'Jane Miller'], 'directors': ['Director B']},
    'Film C': {'genres': ['Action', 'Thriller'], 'regions': ['Asia'], 'actors': ['Charlie Kim', 'Sophie Turner'], 'directors': ['Director C']},
    'Film D': {'genres': ['Drama', 'Mystery'], 'regions': ['Europe'], 'actors': ['David Park', 'Lucas Brown'], 'directors': ['Director D']},
    'Film E': {'genres': ['Sci-Fi', 'Adventure'], 'regions': ['Oceania'], 'actors': ['Eve White', 'Anna Scott'], 'directors': ['Director E']},
    'Film F': {'genres': ['Comedy', 'Family'], 'regions': ['Africa'], 'actors': ['Marcel Dubois', 'Paul Green'], 'directors': ['Director F']},
    'Film G': {'genres': ['Action', 'Adventure'], 'regions': ['South America'], 'actors': ['Nina Lopez', 'Oscar Wilde'], 'directors': ['Director G']},
    'Film H': {'genres': ['Drama', 'Romance'], 'regions': ['Europe'], 'actors': ['Oscar Wilde', 'Marie Curie'], 'directors': ['Director H']},
    'Film I': {'genres': ['Thriller', 'Mystery'], 'regions': ['Asia'], 'actors': ['Paul Green', 'Tina Fey'], 'directors': ['Director I']},
    'Film J': {'genres': ['Sci-Fi', 'Drama'], 'regions': ['North America'], 'actors': ['Quentin Tarantino', 'Rita Ora'], 'directors': ['Director J']},
    'Film K': {'genres': ['Action', 'Comedy'], 'regions': ['Europe'], 'actors': ['Sophie Turner', 'Victor Hugo'], 'directors': ['Director K']},
    'Film L': {'genres': ['Romance', 'Family'], 'regions': ['Africa'], 'actors': ['Tina Fey', 'Wendy Wu'], 'directors': ['Director L']},
    'Film M': {'genres': ['Mystery', 'Adventure'], 'regions': ['Oceania'], 'actors': ['Ursula Andress', 'Xavier Dolan'], 'directors': ['Director M']},
    'Film N': {'genres': ['Sci-Fi', 'Thriller'], 'regions': ['Asia'], 'actors': ['Victor Hugo', 'Yara Shahidi'], 'directors': ['Director N']},
    'Film O': {'genres': ['Action', 'Drama'], 'regions': ['Europe'], 'actors': ['Wendy Wu', 'Zoe Saldana'], 'directors': ['Director O']},
    'Film P': {'genres': ['Comedy', 'Mystery'], 'regions': ['North America'], 'actors': ['Aaron Paul', 'Bella Hadid'], 'directors': ['Director P']},
    'Film Q': {'genres': ['Thriller', 'Adventure'], 'regions': ['South America'], 'actors': ['Cleo de Merode', 'Diana Ross'], 'directors': ['Director Q']},
    'Film R': {'genres': ['Sci-Fi', 'Romance'], 'regions': ['Europe'], 'actors': ['Ethan Hawke', 'Fiona Apple'], 'directors': ['Director R']},
    'Film S': {'genres': ['Drama', 'Family'], 'regions': ['Africa'], 'actors': ['Gina Torres', 'Hugo Weaving'], 'directors': ['Director S']},
    'Film T': {'genres': ['Action', 'Romance'], 'regions': ['Oceania'], 'actors': ['Ivy Chen', 'Jack Black'], 'directors': ['Director T']},
    'Film U': {'genres': ['Comedy', 'Thriller'], 'regions': ['Asia'], 'actors': ['Kurt Russell', 'Lily Collins'], 'directors': ['Director U']},
    'Film V': {'genres': ['Drama', 'Adventure'], 'regions': ['Europe'], 'actors': ['Marcel Dubois', 'Nina Lopez'], 'directors': ['Director V']},
    'Film W': {'genres': ['Sci-Fi', 'Family'], 'regions': ['North America'], 'actors': ['Oscar Wilde', 'Paul Green'], 'directors': ['Director W']},
    'Film X': {'genres': ['Action', 'Mystery'], 'regions': ['Africa'], 'actors': ['Quentin Tarantino', 'Rita Ora'], 'directors': ['Director X']},
    'Film Y': {'genres': ['Romance', 'Thriller'], 'regions': ['Europe'], 'actors': ['Sophie Turner', 'Tina Fey'], 'directors': ['Director Y']},
    'Film Z': {'genres': ['Comedy', 'Sci-Fi'], 'regions': ['Asia'], 'actors': ['Ursula Andress', 'Victor Hugo'], 'directors': ['Director Z']},
    'Film AA': {'genres': ['Adventure', 'Drama'], 'regions': ['Oceania'], 'actors': ['Wendy Wu', 'Xavier Dolan'], 'directors': ['Director AA']},
    'Film AB': {'genres': ['Family', 'Mystery'], 'regions': ['Europe'], 'actors': ['Yara Shahidi', 'Zoe Saldana'], 'directors': ['Director AB']},
    'Film AC': {'genres': ['Thriller', 'Action'], 'regions': ['North America'], 'actors': ['Aaron Paul', 'Bella Hadid'], 'directors': ['Director AC']},
    'Film AD': {'genres': ['Romance', 'Adventure'], 'regions': ['South America'], 'actors': ['Cleo de Merode', 'Diana Ross'], 'directors': ['Director AD']},
    'Film AE': {'genres': ['Sci-Fi', 'Mystery'], 'regions': ['Europe'], 'actors': ['Ethan Hawke', 'Fiona Apple'], 'directors': ['Director AE']},
    'Film AF': {'genres': ['Comedy', 'Drama'], 'regions': ['Africa'], 'actors': ['Gina Torres', 'Hugo Weaving'], 'directors': ['Director AF']},
    'Film AG': {'genres': ['Action', 'Family'], 'regions': ['Oceania'], 'actors': ['Ivy Chen', 'Jack Black'], 'directors': ['Director AG']},
    'Film AH': {'genres': ['Drama', 'Thriller'], 'regions': ['Asia'], 'actors': ['Kurt Russell', 'Lily Collins'], 'directors': ['Director AH']},
    'Film AI': {'genres': ['Sci-Fi', 'Comedy'], 'regions': ['Europe'], 'actors': ['Marcel Dubois', 'Nina Lopez'], 'directors': ['Director AI']},
    'Film AJ': {'genres': ['Mystery', 'Romance'], 'regions': ['North America'], 'actors': ['Oscar Wilde', 'Paul Green'], 'directors': ['Director AJ']},
    'Film AK': {'genres': ['Adventure', 'Action'], 'regions': ['Africa'], 'actors': ['Quentin Tarantino', 'Rita Ora'], 'directors': ['Director AK']},
    'Film AL': {'genres': ['Family', 'Thriller'], 'regions': ['Europe'], 'actors': ['Sophie Turner', 'Tina Fey'], 'directors': ['Director AL']},
    'Film AM': {'genres': ['Comedy', 'Adventure'], 'regions': ['Asia'], 'actors': ['Ursula Andress', 'Victor Hugo'], 'directors': ['Director AM']},
    'Film AN': {'genres': ['Drama', 'Sci-Fi'], 'regions': ['Oceania'], 'actors': ['Wendy Wu', 'Xavier Dolan'], 'directors': ['Director AN']},
    'Film AO': {'genres': ['Thriller', 'Romance'], 'regions': ['Europe'], 'actors': ['Yara Shahidi', 'Zoe Saldana'], 'directors': ['Director AO']},
    'Film AP': {'genres': ['Action', 'Mystery'], 'regions': ['North America'], 'actors': ['Aaron Paul', 'Bella Hadid'], 'directors': ['Director AP']},
    'Film AQ': {'genres': ['Sci-Fi', 'Family'], 'regions': ['South America'], 'actors': ['Cleo de Merode', 'Diana Ross'], 'directors': ['Director AQ']},
    'Film AR': {'genres': ['Comedy', 'Adventure'], 'regions': ['Europe'], 'actors': ['Ethan Hawke', 'Fiona Apple'], 'directors': ['Director AR']},
    'Film AS': {'genres': ['Drama', 'Thriller'], 'regions': ['Africa'], 'actors': ['Gina Torres', 'Hugo Weaving'], 'directors': ['Director AS']},
    'Film AT': {'genres': ['Mystery', 'Romance'], 'regions': ['Oceania'], 'actors': ['Ivy Chen', 'Jack Black'], 'directors': ['Director AT']},
    'Film AU': {'genres': ['Adventure', 'Sci-Fi'], 'regions': ['Asia'], 'actors': ['Kurt Russell', 'Lily Collins'], 'directors': ['Director AU']},
    'Film AV': {'genres': ['Family', 'Comedy'], 'regions': ['Europe'], 'actors': ['Marcel Dubois', 'Nina Lopez'], 'directors': ['Director AV']},
    'Film AW': {'genres': ['Thriller', 'Drama'], 'regions': ['North America'], 'actors': ['Oscar Wilde', 'Paul Green'], 'directors': ['Director AW']},
    'Film AX': {'genres': ['Romance', 'Action'], 'regions': ['Africa'], 'actors': ['Quentin Tarantino', 'Rita Ora'], 'directors': ['Director AX']},
    'Film AY': {'genres': ['Sci-Fi', 'Mystery'], 'regions': ['Europe'], 'actors': ['Sophie Turner', 'Tina Fey'], 'directors': ['Director AY']},
    'Film AZ': {'genres': ['Comedy', 'Thriller'], 'regions': ['Asia'], 'actors': ['Ursula Andress', 'Victor Hugo'], 'directors': ['Director AZ']}
}

attribute_categories = ['genres', 'regions', 'actors', 'directors']
attribute_categories_map = {cat: [] for cat in attribute_categories}
all_attributes = []

for film_data in film_attributes_data_categorized.values():
    for category, items in film_data.items():
        if category in attribute_categories: # Ensure it's one of our defined categories
            for item in items:
                if item not in all_attributes:
                    all_attributes.append(item)
                if item not in attribute_categories_map[category]:
                    attribute_categories_map[category].append(item)

all_attributes.sort()

df_film_attributes_categorized = pd.DataFrame(0, index=film_attributes_data_categorized.keys(), columns=all_attributes)

for film, categories_data in film_attributes_data_categorized.items():
    for category, attributes_list in categories_data.items():
        if category in attribute_categories:
            for attr in attributes_list:
                if attr in df_film_attributes_categorized.columns: # Sanity check
                    df_film_attributes_categorized.loc[film, attr] = 1

print("\n--- DataFrame des Attributs de Films (df_film_attributes_categorized) ---")
print(df_film_attributes_categorized.head())
print(f"\nDimensions du DataFrame des Attributs : {df_film_attributes_categorized.shape}")

print("\n--- Mapping des catégories d'attributs ---")
for cat, attrs in attribute_categories_map.items():
    print(f"- {cat}: {len(attrs)} attributs (ex: {attrs[:3]}...)")

# --- 3. Fonctions Utilitaires (si nécessaires et réutilisables) ---

def cosine_similarity_scratch(vec1, vec2):
    """
    Calcule la similarité Cosinus entre deux vecteurs NumPy.
    Gère les valeurs NaN en les ignorant.
    """
    # Pour les attributs de films, il n'y aura pas de NaN, mais gardons la fonction générique
    # au cas où.
    common_items_mask = ~np.isnan(vec1) & ~np.isnan(vec2)
    vec1_filtered = vec1[common_items_mask]
    vec2_filtered = vec2[common_items_mask]

    if len(vec1_filtered) == 0 or len(vec2_filtered) == 0:
        return 0.0

    dot_product = np.dot(vec1_filtered, vec2_filtered)
    norm_vec1 = np.linalg.norm(vec1_filtered)
    norm_vec2 = np.linalg.norm(vec2_filtered)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (float(norm_vec1) * float(norm_vec2))


def predict_content_based_rating(user_id, movie_to_predict, df_ratings, df_film_attributes,
                                 user_profiles):
    """
    Prédit la note d'un film pour un utilisateur en utilisant le filtrage basé sur le contenu.
    La prédiction est la similarité Cosinus entre le profil de l'utilisateur et les attributs du film.
    """
    if user_id not in user_profiles:
        # S'assurer que le profil de l'utilisateur a déjà été créé
        user_profile = create_user_profile(user_id, df_ratings, df_film_attributes)
        user_profiles[user_id] = user_profile  # Stocker pour réutilisation
    else:
        user_profile = user_profiles[user_id]

    if movie_to_predict not in df_film_attributes.index:
        return np.nan  # Le film n'a pas d'attributs connus

    # Récupérer les attributs du film à prédire
    movie_attributes = df_film_attributes.loc[movie_to_predict]

    # Calculer la similarité Cosinus entre le profil de l'utilisateur et les attributs du film
    # Assurez-vous que user_profile et movie_attributes sont bien des vecteurs numériques (Series ou np.array)

    # Convertir en arrays NumPy pour la fonction cosine_similarity_scratch si ce n'est pas déjà fait
    similarity = cosine_similarity_scratch(user_profile.values, movie_attributes.values)

    # La similarité Cosinus est entre -1 et 1. Nous devons la mapper à une échelle de notes (ex: 1 à 5).
    # Une simple transformation linéaire est (similarity + 1) / 2 * 4 + 1
    # Cela mappe -1 à 1, 0 à 3, et 1 à 5.
    predicted_rating = (similarity + 1) / 2 * 4 + 1

    # On peut limiter la prédiction aux bornes des notes (1 à 5)
    predicted_rating = max(1.0, min(5.0, predicted_rating))

    return predicted_rating

# --- 4. Actions de la phase 4 ---

def create_user_profile(user_id, df_ratings, df_film_attributes):
    """
    Crée un profil de préférences de genre pour un utilisateur donné.
    Le profil est un vecteur où chaque composante représente le poids de l'utilisateur pour un genre donné,
    basé sur les films qu'il a notés. Les notes les plus élevées donnent plus de poids.
    """
    # 1. Sélectionne les notes de l'utilisateur spécifié
    user_ratings = df_ratings.loc[user_id]

    # 2. Filtre les films que l'utilisateur a effectivement notés (non-NaN)
    # et dont la note est supérieure ou égale à 3 (pour filtrer les préférences positives)
    # Tu peux ajuster ce seuil (par exemple, >=4 ou même 5) en fonction de la granularité des préférences.
    rated_positive_films = user_ratings[user_ratings >= 3].index

    # Initialise le profil de l'utilisateur avec des zéros pour chaque genre
    user_profile = pd.Series(0.0, index=df_film_attributes.columns)

    # 3. Pour chaque film noté positivement, ajoute ses attributs au profil de l'utilisateur
    for film_name in rated_positive_films:
        if film_name in df_film_attributes.index:  # Vérifie que le film est bien dans la base d'attributs
            # Pondère les attributs du film par la note de l'utilisateur pour ce film.
            # Plus la note est haute, plus l'influence du genre est forte.
            film_attributes = df_film_attributes.loc[film_name]
            user_profile += film_attributes * user_ratings.loc[film_name]

    # Normaliser le profil si nécessaire (par exemple, pour que la somme des poids soit 1)
    # C'est souvent utile si l'on veut comparer des profils ou si les calculs suivants dépendent de l'échelle.
    # Cependant, pour la similarité cosinus, la normalisation se fait naturellement par la fonction.
    # Pour l'instant, on peut laisser tel quel, ou diviser par la somme pour avoir un profil de probabilités.
    # Si sum est 0 (utilisateur n'a rien noté positivement), éviter la division par zéro.
    if user_profile.sum() > 0:
        user_profile = user_profile / user_profile.sum()

    return user_profile


# --- Test de la fonction pour quelques utilisateurs ---
print("\n--- Profils Utilisateurs basés sur le Contenu ---")
alice_profile = create_user_profile('Alice', df, df_film_attributes)
print("\nProfil d'Alice :\n",
      alice_profile[alice_profile > 0])  # N'affiche que les genres avec un poids > 0

bob_profile = create_user_profile('Bob', df, df_film_attributes)
print("\nProfil de Bob :\n",
      bob_profile[bob_profile > 0])  # N'affiche que les genres avec un poids > 0

# Test avec un utilisateur qui n'a peut-être pas beaucoup noté positivement
charlie_profile = create_user_profile('Charlie', df, df_film_attributes)
print("\nProfil de Charlie :\n", charlie_profile[charlie_profile > 0])

print("\n--- Profils Utilisateurs basés sur le Contenu plus complet ---")
alice_profile = create_user_profile('Alice', df, df_film_attributes2)
print("\nProfil d'Alice :\n",
      alice_profile[alice_profile > 0])  # N'affiche que les genres avec un poids > 0

bob_profile = create_user_profile('Bob', df, df_film_attributes2)
print("\nProfil de Bob :\n",
      bob_profile[bob_profile > 0])  # N'affiche que les genres avec un poids > 0

# Test avec un utilisateur qui n'a peut-être pas beaucoup noté positivement
charlie_profile = create_user_profile('Charlie', df, df_film_attributes2)
print("\nProfil de Charlie :\n", charlie_profile[charlie_profile > 0])


def get_content_based_recommendations(user_id, df_ratings, df_film_attributes, user_profiles,
                                      num_recommendations=5):
    """
    Génère des recommandations de films pour un utilisateur en utilisant le filtrage basé sur le contenu.
    """
    user_rated_movies = df_ratings.loc[user_id].dropna().index.tolist()

    all_movies = df_film_attributes.index.tolist()
    movies_to_predict = [movie for movie in all_movies if movie not in user_rated_movies]

    predictions = {}
    for movie in movies_to_predict:
        predicted_score = predict_content_based_rating(user_id, movie, df_ratings,
                                                       df_film_attributes, user_profiles)
        if not pd.isna(predicted_score):
            predictions[movie] = predicted_score

    # Trier les films par score de prédiction décroissant
    recommended_movies = sorted(predictions.items(), key=lambda item: item[1], reverse=True)

    return recommended_movies[:num_recommendations]

def get_content_based_recommendations_for_user(user_id, df_ratings, df_film_attributes,
                                               num_recommendations=5):
    """
    Fonction principale pour obtenir des recommandations basées sur le contenu pour un utilisateur donné.
    """
    print(
        f"\n--- Génération de recommandations basées sur le contenu pour l'utilisateur : {user_id} ---")

    # 1. Créer ou récupérer le profil de l'utilisateur
    # On utilise un cache simple pour éviter de recréer le profil si la fonction est appelée plusieurs fois
    # dans une même session pour le même utilisateur.
    # Pour un système plus robuste, ce cache pourrait être géré globalement ou passé en argument.
    user_profiles_cache = {}  # Peut être déplacé en dehors de la fonction si on veut un cache persistant
    # pour plusieurs appels successifs sur différents utilisateurs.

    user_profile = create_user_profile(user_id, df_ratings, df_film_attributes)
    user_profiles_cache[user_id] = user_profile

    # Vérifier si le profil de l'utilisateur est vide (pas de notes positives)
    if user_profile.sum() == 0:
        print(
            f"Le profil de l'utilisateur '{user_id}' est vide. Impossible de générer des recommandations basées sur le contenu.")
        print(
            "Cela peut arriver si l'utilisateur n'a pas noté de films ou si toutes ses notes sont inférieures au seuil de préférence (actuellement < 3).")
        return []

    # 2. Obtenir les recommandations en utilisant le profil
    recommendations = get_content_based_recommendations(
        user_id,
        df_ratings,
        df_film_attributes,
        user_profiles_cache,
        # Passer le cache pour que predict_content_based_rating puisse l'utiliser
        num_recommendations=num_recommendations
    )

    if recommendations:
        print(f"\nTop {len(recommendations)} recommandations pour {user_id} :")
        for movie, score in recommendations:
            print(f"- {movie} (Score Prédit : {score:.2f})")
    else:
        print(
            f"Aucune recommandation trouvée pour {user_id}. Tous les films ont peut-être déjà été notés ou aucun film ne correspond au profil.")

    return recommendations


# --- Test du système complet ---
print("\n" + "=" * 50)
print("TEST DU SYSTÈME DE RECOMMANDATION BASÉ SUR LE CONTENU")
print("=" * 50)

# Recommandations pour Alice
recommendations_for_alice = get_content_based_recommendations_for_user('Alice', df,
                                                                       df_film_attributes,
                                                                       num_recommendations=5)

print("\n" + "=" * 50)

# Recommandations pour un autre utilisateur (ex: Bob)
recommendations_for_bob = get_content_based_recommendations_for_user('Bob', df, df_film_attributes,
                                                                     num_recommendations=5)

print("\n" + "=" * 50)

# Test avec un utilisateur qui a peu de notes positives (ou aucune)
# On peut créer un utilisateur fictif ou en choisir un qui a peu noté
# Par exemple, 'David' ou 'Eve' semblent avoir plus de NaN dans le dataset initial.
# Vérifions 'David'
print(f"\nNotes de David:\n{df.loc['David'].dropna()}")
recommendations_for_david = get_content_based_recommendations_for_user('David', df,
                                                                       df_film_attributes,
                                                                       num_recommendations=5)

# --- Test du système complet plus de catégories ---
print("\n" + "=" * 50)
print("TEST DU SYSTÈME DE RECOMMANDATION BASÉ SUR LE CONTENU AMELIORE")
print("=" * 50)

# Recommandations pour Alice
recommendations_for_alice2 = get_content_based_recommendations_for_user('Alice', df,
                                                                       df_film_attributes2,
                                                                       num_recommendations=5)

print("\n" + "=" * 50)

# Recommandations pour un autre utilisateur (ex: Bob)
recommendations_for_bob2 = get_content_based_recommendations_for_user('Bob', df, df_film_attributes2,
                                                                     num_recommendations=5)

print("\n" + "=" * 50)

# Test avec un utilisateur qui a peu de notes positives (ou aucune)
# On peut créer un utilisateur fictif ou en choisir un qui a peu noté
# Par exemple, 'David' ou 'Eve' semblent avoir plus de NaN dans le dataset initial.
# Vérifions 'David'
print(f"\nNotes de David amélioré:\n{df.loc['David'].dropna()}")
recommendations_for_david2 = get_content_based_recommendations_for_user('David', df,
                                                                       df_film_attributes2,
                                                                       num_recommendations=5)

# --- Test du système complet AVEC NOUVELLES CATÉGORIES ---
print("\n" + "=" * 50)
print("TEST DU SYSTÈME DE RECOMMANDATION BASÉ SUR LE CONTENU AVEC CATÉGORIES")
print("=" * 50)

# Recommandations pour Alice
recommendations_for_alice_categorized = get_content_based_recommendations_for_user('Alice', df,
                                                                                df_film_attributes_categorized,
                                                                                num_recommendations=5)

print("\n" + "=" * 50)

# Recommandations pour Bob
recommendations_for_bob_categorized = get_content_based_recommendations_for_user('Bob', df,
                                                                              df_film_attributes_categorized,
                                                                              num_recommendations=5)

print("\n" + "=" * 50)

# Recommandations pour David
print(f"\nNotes de David catégorisées:\n{df.loc['David'].dropna()}")
recommendations_for_david_categorized = get_content_based_recommendations_for_user('David', df,
                                                                                df_film_attributes_categorized,
                                                                                num_recommendations=5)

print("\n--- Détail du profil d'Alice par catégorie ---")
alice_profile_categorized = create_user_profile('Alice', df, df_film_attributes_categorized)

for category, attributes in attribute_categories_map.items():
    # Sélectionne uniquement les attributs de cette catégorie dans le profil d'Alice
    category_profile = alice_profile_categorized[attributes]
    # N'affiche que les valeurs positives pour plus de clarté
    if category_profile[category_profile > 0].empty:
        print(f"\n{category.capitalize()} (Alice): Aucune préférence positive détectée.")
    else:
        print(f"\n{category.capitalize()} (Alice) :")
        # Trie pour un affichage plus propre
        print(category_profile[category_profile > 0].sort_values(ascending=False))
