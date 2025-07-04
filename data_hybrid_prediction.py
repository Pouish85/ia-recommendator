import pandas as pd
import numpy as np
import random

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

def predict_user_based(user_to_predict, movie_to_predict, ratings_df, user_similarity_matrix,
                       user_averages):
    """
    Prédit la note d'un utilisateur pour un film non noté en utilisant le filtrage collaboratif User-Based.

    :param user_to_predict: Nom de l'utilisateur pour lequel prédire.
    :param movie_to_predict: Nom du film pour lequel prédire la note.
    :param ratings_df: DataFrame complet des notes.
    :param user_similarity_matrix: Matrice de similarité User-User.
    :param user_averages: Series des notes moyennes par utilisateur.
    :return: Note prédite (float) ou np.nan si impossible de prédire.
    """

    # 1. Récupérer la note moyenne de l'utilisateur à prédire
    # Gérer le cas où l'utilisateur n'est pas dans les moyennes (peu probable si bien initialisé)
    if user_to_predict not in user_averages:
        return np.nan  # Ou soulever une erreur

    user_mean_rating = user_averages.loc[user_to_predict]

    # 2. Initialiser les sommes pour la formule
    numerator = 0.0
    denominator = 0.0

    # 3. Parcourir les voisins potentiels (tous les autres utilisateurs)
    # Exclure l'utilisateur à prédire lui-même

    # Récupérer les similarités de l'utilisateur avec tous les autres
    similarities_with_user = user_similarity_matrix.loc[user_to_predict]

    for neighbor_user in ratings_df.index:  # Itérer sur tous les utilisateurs
        if neighbor_user == user_to_predict:
            continue  # Ne pas se comparer à soi-même

        # Vérifier si le voisin a noté le film que l'on veut prédire
        neighbor_rating_for_movie = ratings_df.loc[neighbor_user, movie_to_predict]

        if not pd.isna(neighbor_rating_for_movie):
            # C'est un voisin valide qui a noté le film

            # Récupérer la similarité entre l'utilisateur et le voisin
            similarity = similarities_with_user.loc[neighbor_user]

            # Gérer le cas où la similarité est NaN (pas de films communs)
            if pd.isna(similarity):
                continue

            # Récupérer la note moyenne du voisin
            neighbor_mean_rating = user_averages.loc[neighbor_user]

            # Ajouter à la somme pondérée
            numerator += similarity * (neighbor_rating_for_movie - neighbor_mean_rating)
            denominator += abs(similarity)

    # 4. Calculer la note prédite
    if denominator == 0:
        return np.nan  # Impossible de prédire si pas de voisins valides ou similarités nulles

    predicted_rating = user_mean_rating + (numerator / denominator)

    # Optionnel: Clamper la note prédite entre les bornes de ton échelle de notes (ex: 1 à 5)
    predicted_rating = max(1.0, min(5.0, predicted_rating))

    return predicted_rating

def predict_item_based(user_to_predict, movie_to_predict, ratings_df, item_similarity_matrix):
    """
    Prédit la note d'un utilisateur pour un film non noté en utilisant le filtrage collaboratif Item-Based.

    :param user_to_predict: Nom de l'utilisateur pour lequel prédire.
    :param movie_to_predict: Nom du film pour lequel prédire la note.
    :param ratings_df: DataFrame complet des notes.
    :param item_similarity_matrix: Matrice de similarité Item-Item.
    :return: Note prédite (float) ou np.nan si impossible de prédire.
    """

    numerator = 0.0
    denominator = 0.0

    # Récupérer la ligne de similarités pour le film à prédire (avec tous les autres films)
    similarities_with_movie = item_similarity_matrix.loc[movie_to_predict]

    # Parcourir tous les films que l'utilisateur a déjà notés
    # On itère sur les colonnes du DataFrame (qui sont les films)
    for rated_movie in ratings_df.columns:
        # Vérifier si l'utilisateur a noté ce film 'rated_movie'
        user_rating_for_rated_movie = ratings_df.loc[user_to_predict, rated_movie]

        # S'assurer que l'utilisateur a bien noté ce film ET que ce n'est pas le film que l'on veut prédire
        if not pd.isna(user_rating_for_rated_movie) and rated_movie != movie_to_predict:

            # Récupérer la similarité entre le film à prédire et ce film 'rated_movie'
            similarity = similarities_with_movie.loc[rated_movie]

            # Gérer le cas où la similarité est NaN (pas de notes communes pour les films)
            if pd.isna(similarity):
                continue

            # Ajouter à la somme pondérée
            numerator += similarity * user_rating_for_rated_movie
            denominator += abs(similarity)

    # Calculer la note prédite
    if denominator == 0:
        return np.nan  # Impossible de prédire si pas de films similaires notés par l'utilisateur

    predicted_rating = numerator / denominator

    # Optionnel: Clamper la note prédite entre les bornes de ton échelle de notes (ex: 1 à 5)
    predicted_rating = max(1.0, min(5.0, predicted_rating))

    return predicted_rating

def calculate_user_averages(ratings_df):
    """
    Calcule la note moyenne de chaque utilisateur, en ignorant les NaN.

    :param ratings_df: DataFrame des notes (utilisateurs en index, films en colonnes)
    :return: Series Pandas avec les moyennes de notes par utilisateur
    """
    # axis=1 signifie que l'on calcule la moyenne sur les lignes (par utilisateur)
    # skipna=True est le comportement par défaut pour .mean(), il ignore les NaN
    return ratings_df.mean(axis=1)

def cosine_similarity_scratch(vec1, vec2):
    """
    Calcule la similarité Cosinus entre deux vecteurs NumPy.
    Gère les valeurs NaN en les ignorant.
    """
    # Masque pour les valeurs non NaN (présentes dans les deux vecteurs)
    # Utilisez une copie pour éviter les SettingWithCopyWarning
    common_items_mask = ~np.isnan(vec1) & ~np.isnan(vec2)

    # Filtrer les vecteurs pour ne garder que les éléments communs non NaN
    vec1_filtered = vec1[common_items_mask]
    vec2_filtered = vec2[common_items_mask]

    # Si aucun élément commun, la similarité est indéfinie (ou 0 par convention)
    if len(vec1_filtered) == 0 or len(vec2_filtered) == 0:
        return 0.0  # Ou np.nan si vous préférez

    dot_product = np.dot(vec1_filtered, vec2_filtered)
    norm_vec1 = np.linalg.norm(vec1_filtered)
    norm_vec2 = np.linalg.norm(vec2_filtered)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Éviter la division par zéro

    # noinspection PyTypeChecker
    return dot_product / (norm_vec1 * norm_vec2)

    return 1 / (1 + distance)

# --- 4. Hybrid Prediction ---

# Notes moyennes des utilisateurs
user_averages = calculate_user_averages(df)
print("\n--- Notes moyennes par utilisateur (user_averages) ---")
print(user_averages.head())

# Matrice de similarité utilisateur-utilisateur (Cosine)
user_similarity_matrix_cosine = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
for user1 in df.index:
    for user2 in df.index:
        if user1 == user2:
            user_similarity_matrix_cosine.loc[user1, user2] = 1.0
        else:
            vec1 = df.loc[user1].values
            vec2 = df.loc[user2].values
            user_similarity_matrix_cosine.loc[user1, user2] = cosine_similarity_scratch(vec1, vec2)

print("\n--- Matrice de Similarité User-User (Cosine) ---")
print(user_similarity_matrix_cosine.head())

item_similarity_matrix_cosine = pd.DataFrame(index=df.index,
                                                   columns=df.index, dtype=float)
for item1 in df.index:
    for item2 in df.index:
        if item1 == item2:
            item_similarity_matrix_cosine.loc[item1, item2] = 1.0
        else:
            vec1 = df.loc[item1].values
            vec2 = df.loc[item2].values
            item_similarity_matrix_cosine.loc[item1, item2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (Item-Item, sur df_train) :")
print(item_similarity_matrix_cosine.round(4))

# Matrice de similarité item-item (Cosine)
item_similarity_matrix_cosine = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
for film1 in df.columns:
    for film2 in df.columns:
        if film1 == film2:
            item_similarity_matrix_cosine.loc[film1, film2] = 1.0
        else:
            vec1 = df[film1].values
            vec2 = df[film2].values
            item_similarity_matrix_cosine.loc[film1, film2] = cosine_similarity_scratch(vec1, vec2)

print("\n--- Matrice de Similarité Item-Item (Cosine) ---")
print(item_similarity_matrix_cosine.head())


# --- Fonctions pour la Phase 5 : Hybrid Recommender Engine ---

def predict_hybrid_rating(user, movie, df_notes, df_film_attributes_categorized,
                          user_similarity_matrix, item_similarity_matrix_cosine, user_averages,
                          weight_content_based=0.5, weight_collaborative=0.5,
                          user_profiles_cache_hybrid={}):  # Cache for hybrid, local to function scope
    """
    Prédit la note d'un film pour un utilisateur en combinant les approches
    Content-Based et Collaborative Filtering (User-Based).

    :param user: Nom de l'utilisateur.
    :param movie: Nom du film.
    :param df_notes: DataFrame des notes (utilisateurs en index, films en colonnes).
    :param df_film_attributes_categorized: DataFrame des attributs de films (films en index, catégories en colonnes).
    :param user_similarity_matrix: Matrice de similarité utilisateur-utilisateur.
    :param item_similarity_matrix_cosine: Matrice de similarité item-item (Cosine).
    :param user_averages: Series des notes moyennes par utilisateur.
    :param weight_content_based: Poids à appliquer à la prédiction Content-Based.
    :param weight_collaborative: Poids à appliquer à la prédiction Collaborative.
    :param user_profiles_cache_hybrid: Cache pour les profils utilisateurs afin d'éviter les recalculs.
    :return: Note hybride prédite pour le film par l'utilisateur, ou np.nan si impossible.
    """

    # Tenter de récupérer les prédictions des deux systèmes
    content_pred = predict_content_based_rating(user, movie, df_notes,
                                                df_film_attributes_categorized,
                                                user_profiles_cache_hybrid)
    # collaborative_pred = predict_user_based(user, movie, df_notes, user_similarity_matrix,
    #                                         user_averages)
    collaborative_pred = predict_item_based(user, movie, df_notes, item_similarity_matrix_cosine)
    # Note: On utilise predict_user_based pour le collaboratif ici, vous pourriez choisir item-based ou combiner les deux aussi.

    # Gérer les cas où l'une des prédictions n'est pas disponible
    if pd.isna(content_pred) and pd.isna(collaborative_pred):
        return np.nan  # Aucune prédiction n'est possible
    elif pd.isna(content_pred):
        # Si seul le collaboratif est disponible, ne prendre que celui-ci
        return collaborative_pred
    elif pd.isna(collaborative_pred):
        # Si seul le content-based est disponible, ne prendre que celui-ci
        return content_pred
    else:
        # Combiner les prédictions avec les poids
        # Le dénominateur assure que même si la somme des poids n'est pas 1, la note finale est bien mise à l'échelle.
        hybrid_pred = (
                                  weight_content_based * content_pred + weight_collaborative * collaborative_pred) / (
                                  weight_content_based + weight_collaborative)
        # S'assurer que la note prédite est dans les limites de l'échelle (ex: 1 à 5)
        return max(1.0, min(5.0, hybrid_pred))


def get_hybrid_recommendations(user, df_notes, df_film_attributes_categorized,
                               user_similarity_matrix, user_averages,
                               num_recommendations=5, weight_content_based=0.5,
                               weight_collaborative=0.5):
    """
    Génère des recommandations hybrides pour un utilisateur.

    :param user: Nom de l'utilisateur.
    :param df_notes: DataFrame des notes.
    :param df_film_attributes_categorized: DataFrame des attributs de films.
    :param user_similarity_matrix: Matrice de similarité utilisateur-utilisateur.
    :param user_averages: Series des notes moyennes par utilisateur.
    :param num_recommendations: Nombre de recommandations à retourner.
    :param weight_content_based: Poids pour le content-based.
    :param weight_collaborative: Poids pour le collaboratif.
    :return: Liste de tuples (film, score prédit) triée par score décroissant.
    """

    predicted_ratings = []

    # Identifier les films que l'utilisateur n'a pas encore notés
    if user not in df_notes.index:
        print(f"Utilisateur '{user}' non trouvé dans le dataset de notes.")
        return []

    movies_to_predict = df_notes.columns[df_notes.loc[user].isna()].tolist()

    # Cache des profils utilisateurs, pour éviter de le recréer à chaque appel à predict_content_based_rating
    user_profiles_cache_for_recs = {}

    for movie in movies_to_predict:
        # Utiliser la fonction de prédiction hybride
        pred_score = predict_hybrid_rating(user, movie, df_notes, df_film_attributes_categorized,
                                           user_similarity_matrix, item_similarity_matrix_cosine, user_averages,
                                           weight_content_based, weight_collaborative,
                                           user_profiles_cache_for_recs)

        if not pd.isna(pred_score):
            predicted_ratings.append((movie, pred_score))

    # Trier les films par score prédit décroissant
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    return predicted_ratings[:num_recommendations]


# --- Exemple d'utilisation dans data_hybrid_prediction.py ---
if __name__ == '__main__':

    # Assurez-vous que df, df_film_attributes_categorized, user_similarity_matrix_cosine, user_averages
    # sont disponibles ici. (Ils sont calculés ci-dessus dans ce fichier).

    # Exemple de prédiction de note hybride pour un film spécifique
    user_example = 'Alice'
    movie_example = 'Film D'  # Film non noté par Alice dans le dataset initial

    print(f"\nPrédiction hybride pour {user_example} sur '{movie_example}':")
    hybrid_predicted_rating = predict_hybrid_rating(
        user_example, movie_example, df, df_film_attributes_categorized,
        user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
        weight_content_based=0.6, weight_collaborative=0.4
    )
    if not pd.isna(hybrid_predicted_rating):
        print(f"Note hybride prédite : {hybrid_predicted_rating:.4f}")
    else:
        print(f"Impossible de prédire une note hybride pour {user_example} sur '{movie_example}'.")

    # Exemple de recommandations hybrides pour Alice
    print("\nRecommandations Hybrides pour Alice (poids 0.5 Content, 0.5 Collaboratif) :")
    hybrid_recs_alice = get_hybrid_recommendations(
        'Alice', df, df_film_attributes_categorized,
        user_similarity_matrix_cosine, user_averages,
        num_recommendations=5
    )

    if hybrid_recs_alice:
        for movie, score in hybrid_recs_alice:
            print(f"- {movie}: {score:.4f}")
    else:
        print("Aucune recommandation hybride trouvée pour Alice.")

    # Exemple de recommandations hybrides pour David, en privilégiant le Content-Based
    print("\nRecommandations Hybrides pour David (poids 0.7 Content, 0.3 Collaboratif) :")
    hybrid_recs_david = get_hybrid_recommendations(
        'David', df, df_film_attributes_categorized,
        user_similarity_matrix_cosine, user_averages,
        num_recommendations=5, weight_content_based=0.7, weight_collaborative=0.3
    )

    if hybrid_recs_david:
        for movie, score in hybrid_recs_david:
            print(f"- {movie}: {score:.4f}")
    else:
        print("Aucune recommandation hybride trouvée pour David.")

# --- Evaluation des recommandations ---

df_train = df.copy()
test_set = []

random.seed(42)
num_items_to_hide = 1
for user in df_train.index:
    user_rated_movies = df_train.loc[user].dropna().index.tolist()

    if len(user_rated_movies) > num_items_to_hide:
        movies_to_hide = random.sample(user_rated_movies, num_items_to_hide)

        for movie in movies_to_hide:
            original_rating = df_train.loc[user, movie]
            test_set.append({'user': user, 'movie': movie, 'original_rating': original_rating})
            df_train.loc[user, movie] = np.nan

print(f"\nDataFrame d'entraînement (notes cachées) :\n{df_train}")
print(f"\nEnsemble de test (notes originales cachées) :\n{pd.DataFrame(test_set)}")

# IMPORTANT : Recalculer les matrices de similarité et les moyennes avec df_train
user_averages_train = calculate_user_averages(df_train)
print("\n--- Notes moyennes par utilisateur (avec df_train) ---")
print(user_averages_train)

user_similarity_matrix_cosine_train = pd.DataFrame(index=df_train.index, columns=df_train.index,
                                                   dtype=float)
for user1 in df_train.index:
    for user2 in df_train.index:
        if user1 == user2:
            user_similarity_matrix_cosine_train.loc[user1, user2] = 1.0
        else:
            vec1 = df_train.loc[user1].values
            vec2 = df_train.loc[user2].values
            user_similarity_matrix_cosine_train.loc[user1, user2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (User-User, sur df_train) :")
print(user_similarity_matrix_cosine_train.round(4))

item_similarity_matrix_cosine_train = pd.DataFrame(index=df_train.index,
                                                   columns=df_train.index, dtype=float)
for item1 in df_train.index:
    for item2 in df_train.index:
        if item1 == item2:
            item_similarity_matrix_cosine_train.loc[item1, item2] = 1.0
        else:
            vec1 = df_train.loc[item1].values
            vec2 = df_train.loc[item2].values
            item_similarity_matrix_cosine_train.loc[item1, item2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (Item-Item, sur df_train) :")
print(item_similarity_matrix_cosine_train.round(4))

df_train_transposed = df_train.T
item_similarity_matrix_cosine_train = pd.DataFrame(index=df_train_transposed.index,
                                                   columns=df_train_transposed.index, dtype=float)
for item1 in df_train_transposed.index:
    for item2 in df_train_transposed.index:
        if item1 == item2:
            item_similarity_matrix_cosine_train.loc[item1, item2] = 1.0
        else:
            vec1 = df_train_transposed.loc[item1].values
            vec2 = df_train_transposed.loc[item2].values
            item_similarity_matrix_cosine_train.loc[item1, item2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (Item-Item, sur df_train) :")
print(item_similarity_matrix_cosine_train.round(4))

def evaluate_recommender(predictions, actual_ratings):
    """
    Calcule la Mean Absolute Error (MAE) et la Root Mean Squared Error (RMSE).
    """
    if not predictions or not actual_ratings:
        return np.nan, np.nan

    if len(predictions) != len(actual_ratings):
        raise ValueError(
            "Les listes de prédictions et de notes réelles doivent avoir la même taille.")

    predictions_np = np.array(predictions)
    actual_ratings_np = np.array(actual_ratings)

    abs_errors = np.abs(predictions_np - actual_ratings_np)
    mae = np.mean(abs_errors)

    squared_errors = (predictions_np - actual_ratings_np) ** 2
    rmse = np.sqrt(np.mean(squared_errors))

    return mae, rmse

user_based_predictions = []
item_based_predictions = []
user_based_actual_ratings = []
item_based_actual_ratings = []
hybrid_predictions55 = []
hybrid_actual_ratings55 = []
hybrid_predictions73 = []
hybrid_actual_ratings73 = []
hybrid_predictions37 = []
hybrid_actual_ratings37 = []
hybrid_predictions28 = []
hybrid_actual_ratings28 = []
hybrid_predictions19 = []
hybrid_actual_ratings19 = []

for entry in test_set:
    user = entry['user']
    movie = entry['movie']
    original_rating = entry['original_rating']

    # Prédiction User-Based
    predicted_user_based = predict_user_based(user, movie, df_train,
                                              user_similarity_matrix_cosine_train,
                                              user_averages_train)
    if not pd.isna(predicted_user_based):
        user_based_predictions.append(predicted_user_based)
        user_based_actual_ratings.append(original_rating)

    # Prédiction Item-Based
    predicted_item_based = predict_item_based(user, movie, df_train,
                                              item_similarity_matrix_cosine_train)
    if not pd.isna(predicted_item_based):
        item_based_predictions.append(predicted_item_based)
        item_based_actual_ratings.append(original_rating)

    # Prédiction Hybride 50% Content-Based, 50% Collaborative
    predicted_hybrid55 = predict_hybrid_rating(
        user, movie, df_train, df_film_attributes_categorized,
        # <-- df_film_attributes_categorized doit être défini
        user_similarity_matrix_cosine_train, item_similarity_matrix_cosine_train, user_averages_train,
        weight_content_based=0.5, weight_collaborative=0.5
    )

    if not pd.isna(predicted_hybrid55):
        hybrid_predictions55.append(predicted_hybrid55)
        hybrid_actual_ratings55.append(original_rating)

    # Prédiction Hybride 70% Content-Based, 30% Collaborative
    predicted_hybrid73 = predict_hybrid_rating(
        user, movie, df_train, df_film_attributes_categorized,
        # <-- df_film_attributes_categorized doit être défini
        user_similarity_matrix_cosine_train, item_similarity_matrix_cosine_train, user_averages_train,
        weight_content_based=0.7, weight_collaborative=0.3
    )

    if not pd.isna(predicted_hybrid73):
        hybrid_predictions73.append(predicted_hybrid73)
        hybrid_actual_ratings73.append(original_rating)

    # Prédiction Hybride 30% Content-Based, 70% Collaborative
    predicted_hybrid37 = predict_hybrid_rating(
        user, movie, df_train, df_film_attributes_categorized,
        # <-- df_film_attributes_categorized doit être défini
        user_similarity_matrix_cosine_train, item_similarity_matrix_cosine_train, user_averages_train,
        weight_content_based=0.3, weight_collaborative=0.7
    )

    if not pd.isna(predicted_hybrid37):
        hybrid_predictions37.append(predicted_hybrid37)
        hybrid_actual_ratings37.append(original_rating)

    # Prédiction Hybride 20% Content-Based, 80% Collaborative
    predicted_hybrid28 = predict_hybrid_rating(
        user, movie, df_train, df_film_attributes_categorized,
        # <-- df_film_attributes_categorized doit être défini
        user_similarity_matrix_cosine_train, item_similarity_matrix_cosine_train,
        user_averages_train,
        weight_content_based=0.2, weight_collaborative=0.8
    )

    if not pd.isna(predicted_hybrid28):
        hybrid_predictions28.append(predicted_hybrid28)
        hybrid_actual_ratings28.append(original_rating)

    # Prédiction Hybride 10% Content-Based, 90% Collaborative
    predicted_hybrid19 = predict_hybrid_rating(
        user, movie, df_train, df_film_attributes_categorized,
        # <-- df_film_attributes_categorized doit être défini
        user_similarity_matrix_cosine_train, item_similarity_matrix_cosine_train,
        user_averages_train,
        weight_content_based=0.1, weight_collaborative=0.9
    )

    if not pd.isna(predicted_hybrid19):
        hybrid_predictions19.append(predicted_hybrid19)
        hybrid_actual_ratings19.append(original_rating)

print("\n--- Résultats d'Évaluation ---")

# Évaluation User-Based
if user_based_predictions:
    mae_ub, rmse_ub = evaluate_recommender(user_based_predictions, user_based_actual_ratings)
    print(f"User-Based Recommender (N={len(user_based_predictions)} prédictions) :")
    print(f"  MAE : {mae_ub:.4f}")
    print(f"  RMSE : {rmse_ub:.4f}")
else:
    print("Pas assez de prédictions User-Based pour l'évaluation.")

# Évaluation Item-Based
if item_based_predictions:
    mae_ib, rmse_ib = evaluate_recommender(item_based_predictions, item_based_actual_ratings)
    print(f"Item-Based Recommender (N={len(item_based_predictions)} prédictions) :")
    print(f"  MAE : {mae_ib:.4f}")
    print(f"  RMSE : {rmse_ib:.4f}")
else:
    print("Pas assez de prédictions Item-Based pour l'évaluation.")

# Évaluation Hybride 50/50
if hybrid_predictions55:
    mae_hybrid, rmse_hybrid = evaluate_recommender(hybrid_predictions55, hybrid_actual_ratings55)
    print(f"Hybrid Recommender 50/50 (N={len(hybrid_predictions55)} prédictions) :")
    print(f"  MAE : {mae_hybrid:.4f}")
    print(f"  RMSE : {rmse_hybrid:.4f}")
else:
    print("Pas assez de prédictions Hybrides pour l'évaluation.")

# Évaluation Hybride 70/30
if hybrid_predictions73:
    mae_hybrid, rmse_hybrid = evaluate_recommender(hybrid_predictions73, hybrid_actual_ratings73)
    print(f"Hybrid Recommender 70/30 (N={len(hybrid_predictions73)} prédictions) :")
    print(f"  MAE : {mae_hybrid:.4f}")
    print(f"  RMSE : {rmse_hybrid:.4f}")
else:
    print("Pas assez de prédictions Hybrides pour l'évaluation.")

# Évaluation Hybride 30/70
if hybrid_predictions37:
    mae_hybrid, rmse_hybrid = evaluate_recommender(hybrid_predictions37, hybrid_actual_ratings37)
    print(f"Hybrid Recommender 30/70 (N={len(hybrid_predictions37)} prédictions) :")
    print(f"  MAE : {mae_hybrid:.4f}")
    print(f"  RMSE : {rmse_hybrid:.4f}")
else:
    print("Pas assez de prédictions Hybrides pour l'évaluation.")

# Évaluation Hybride 20/80
if hybrid_predictions28:
    mae_hybrid, rmse_hybrid = evaluate_recommender(hybrid_predictions28, hybrid_actual_ratings28)
    print(f"Hybrid Recommender 20/80 (N={len(hybrid_predictions28)} prédictions) :")
    print(f"  MAE : {mae_hybrid:.4f}")
    print(f"  RMSE : {rmse_hybrid:.4f}")
else:
    print("Pas assez de prédictions Hybrides pour l'évaluation.")

# Évaluation Hybride 10/90
if hybrid_predictions19:
    mae_hybrid, rmse_hybrid = evaluate_recommender(hybrid_predictions19, hybrid_actual_ratings19)
    print(f"Hybrid Recommender 10/90 (N={len(hybrid_predictions19)} prédictions) :")
    print(f"  MAE : {mae_hybrid:.4f}")
    print(f"  RMSE : {rmse_hybrid:.4f}")
else:
    print("Pas assez de prédictions Hybrides pour l'évaluation.")
