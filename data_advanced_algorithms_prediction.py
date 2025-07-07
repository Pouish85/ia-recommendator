import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split

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

# --- 3. Gérer les NaN pour le calcul de similarité ---

# --- 3a. Remplacer les NaN par 0 pour le calcul de similarité ---

df_filled = df.fillna(0)
print("\nDataFrame avec NaN remplacés par 0 :\n", df_filled)

# --- 3B autre méthode que remplacer les NaN par 0 ---

# --- TODO : Implémenter une méthode alternative pour gérer les NaN ---

# --- 4. Calcul de la Matrice de Similarité User-User ---

# --- 4a. Avec nan = 0 ---

user_similarity_matrix_sklearn = cosine_similarity(df_filled)
user_similarity_df_sklearn = pd.DataFrame(user_similarity_matrix_sklearn,
                                          index=df.index, columns=df.index)

print("\nMatrice de Similarité User-User (Scikit-learn - Cosinus) :\n",
      user_similarity_df_sklearn.round(4))

# --- 4a. Avec autre methode ---

# --- 5. Calcul de la Matrice de Similarité Item-Item ---

# --- 5a. Avec nan = 0 ---

item_similarity_matrix_sklearn = cosine_similarity(df_filled.T)
item_similarity_df_sklearn = pd.DataFrame(item_similarity_matrix_sklearn,
                                          index=df.columns, columns=df.columns)

print("\nMatrice de Similarité Item-Item (Scikit-learn - Cosinus) :\n",
      item_similarity_df_sklearn.round(4))

# --- 5a. Avec autre methode ---


# --- 6. Décomposition de Matrice avec TruncatedSVD ---

print("\n--- Décomposition de Matrice avec TruncatedSVD ---")

n_components = 5
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_latent_features_matrix = svd.fit_transform(df_filled)
movie_latent_features_matrix = svd.components_.T

print(f"\nMatrice des Caractéristiques Latentes Utilisateur (dimensions: {user_latent_features_matrix.shape}) :")
print(user_latent_features_matrix[:5].round(4))

print(f"\nMatrice des Caractéristiques Latentes Film (dimensions: {movie_latent_features_matrix.shape}) :")
print(movie_latent_features_matrix[:5].round(4))

user_latent_features_df = pd.DataFrame(user_latent_features_matrix,
                                       index=df.index,
                                       columns=[f'latent_f{i+1}' for i in range(n_components)])

movie_latent_features_df = pd.DataFrame(movie_latent_features_matrix,
                                        index=df.columns,
                                        columns=[f'latent_f{i+1}' for i in range(n_components)])

print("\nDataFrame des Caractéristiques Latentes Utilisateur (premières lignes) :")
print(user_latent_features_df.head())

print("\nDataFrame des Caractéristiques Latentes Film (premières lignes) :")
print(movie_latent_features_df.head())

# Exemple : Prédiction de la note d'Alice pour le Film D
# (Attention: TruncatedSVD ici est pour la réduction de dimensionnalité, pas une prédiction directe)
# Pour une prédiction réelle, il faudrait multiplier les matrices originales, ou utiliser un modèle plus adapté.

# C'est principalement pour montrer comment les "composantes" sont générées.
# Pour la prédiction, on reconstruirait la matrice :
reconstructed_matrix = np.dot(user_latent_features_matrix, movie_latent_features_matrix.T)
print("\nMatrice de notes reconstruite (exemple, via TruncatedSVD) :\n",
      pd.DataFrame(reconstructed_matrix, index=df.index, columns=df.columns).round(2))

# --- 7. Utilisation de la bibliothèque Surprise pour la prédiction ---

print("\n--- Utilisation de la bibliothèque Surprise pour la prédiction ---")

# --- Préparation du dataset pour Surprise ---
# Surprise a besoin des données dans un format "long" : [utilisateur, élément, note]
ratings_long = df.stack().reset_index()
ratings_long.columns = ['user', 'item', 'rating']

# Un Reader est nécessaire pour que Surprise comprenne la plage de notes.
# Ici, les notes sont de 1 à 5.
reader = Reader(rating_scale=(1, 5))

# Charger le dataset à partir du DataFrame Pandas
data_surprise = Dataset.load_from_df(ratings_long[['user', 'item', 'rating']], reader)

# --- Entraînement et Évaluation du modèle SVD avec Surprise ---
print("\n--- Entraînement et Évaluation du modèle SVD avec Surprise (Cross-Validation) ---")

# Nous allons utiliser la validation croisée (cv=5) pour évaluer la performance
# de manière robuste. RMSE et MAE sont les métriques les plus courantes.
# n_epochs est le nombre d'itérations de l'algorithme de descente de gradient.
# random_state=42 pour la reproductibilité.
results = cross_validate(SVD(random_state=42, n_epochs=20), data_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\n--- Résultats de la Cross-Validation (SVD) ---")
print(f"RMSE moyen sur 5 folds : {results['test_rmse'].mean():.4f}")
print(f"MAE moyen sur 5 folds : {results['test_mae'].mean():.4f}")

print("\n--- Entraînement du modèle SVD sur l'ensemble du dataset ---")
# build_full_trainset() construit un jeu d'entraînement à partir de toutes les données.
trainset = data_surprise.build_full_trainset()
algo = SVD(random_state=42, n_epochs=20) # Utilisation des mêmes paramètres que pour l'évaluation
algo.fit(trainset)
print("Modèle SVD entraîné avec succès.")

# --- Générer des recommandations pour un utilisateur spécifique (par exemple, 'Alice') ---
user_id = 'Alice'
print(f"\n--- Recommandations pour l'utilisateur : {user_id} ---")

# Obtenir la liste des films déjà notés par cet utilisateur
movies_rated_by_user = df.loc[user_id].dropna().index.tolist()
print(f"Films déjà notés par {user_id} : {movies_rated_by_user}")

# Obtenir la liste de tous les films
all_movies = df.columns.tolist()

# Identifier les films non encore notés par l'utilisateur
movies_to_predict = [movie for movie in all_movies if movie not in movies_rated_by_user]
print(f"Films à prédire pour {user_id} (non notés) : {movies_to_predict}")

# Faire des prédictions pour ces films non notés
predictions = []
for movie_id in movies_to_predict:
    # La méthode predict() prend (user_id, item_id, r_ui_true=None, clip=True, verbose=False)
    # r_ui_true est la vraie note (que nous ne connaissons pas ici pour les prédictions)
    # clip=True garantit que la prédiction est dans la plage de notes (1 à 5)
    prediction = algo.predict(user_id, movie_id)
    predictions.append((movie_id, prediction.est)) # prediction.est est la note prédite

# Trier les prédictions par note prédite décroissante
predictions.sort(key=lambda x: x[1], reverse=True)

# Afficher les top N recommandations
top_n = 5 # Nombre de recommandations à afficher
print(f"\nTop {top_n} recommandations pour {user_id} :")
for movie, predicted_rating in predictions[:top_n]:
    print(f"- {movie}: {predicted_rating:.4f}")

# --- Exemple pour un autre utilisateur, 'David' ---
user_id = 'David'
print(f"\n--- Recommandations pour l'utilisateur : {user_id} ---")

movies_rated_by_user = df.loc[user_id].dropna().index.tolist()
print(f"Films déjà notés par {user_id} : {movies_rated_by_user}")

movies_to_predict = [movie for movie in all_movies if movie not in movies_rated_by_user]
print(f"Films à prédire pour {user_id} (non notés) : {movies_to_predict}")

predictions = []
for movie_id in movies_to_predict:
    prediction = algo.predict(user_id, movie_id)
    predictions.append((movie_id, prediction.est))

predictions.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop {top_n} recommandations pour {user_id} :")
for movie, predicted_rating in predictions[:top_n]:
    print(f"- {movie}: {predicted_rating:.4f}")
