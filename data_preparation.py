import pandas as pd
import numpy as np

# Dataset simple : Utilisateurs (lignes) vs. Films/Produits (colonnes)
# Les valeurs représentent les notes (par exemple, de 1 à 5)
data = {
    'Film A': [5, 4, np.nan, 1, 3, 4],
    'Film B': [3, np.nan, 4, 2, 5, 4],
    'Film C': [4, 5, 2, np.nan, 4, 4],
    'Film D': [np.nan, 3, 5, 4, 2, 4]
}

# Créez le DataFrame Pandas avec des noms d'utilisateurs comme index
index = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Marcel']
df = pd.DataFrame(data, index=index)

print("Notre dataset initial :")
print(df)


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


# Exemple d'utilisation de la similarité Cosinus
print("\n--- Similarité Cosinus ---")
# Comparons Alice et Bob
alice_ratings = df.loc['Alice'].values
bob_ratings = df.loc['Bob'].values
sim_alice_bob = cosine_similarity_scratch(alice_ratings, bob_ratings)
print(f"Similarité Cosinus entre Alice et Bob : {sim_alice_bob:.4f}")

# Comparons Charlie et David
charlie_ratings = df.loc['Charlie'].values
david_ratings = df.loc['David'].values
sim_charlie_david = cosine_similarity_scratch(charlie_ratings, david_ratings)
print(f"Similarité Cosinus entre Charlie et David : {sim_charlie_david:.4f}")

# Calculons la matrice de similarité utilisateur (user-user)
user_similarity_matrix_cosine = pd.DataFrame(index=df.index, columns=df.index, dtype=float)

for user1 in df.index:
    for user2 in df.index:
        if user1 == user2:
            user_similarity_matrix_cosine.loc[
                user1, user2] = 1.0  # Un utilisateur est parfaitement similaire à lui-même
        else:
            vec1 = df.loc[user1].values
            vec2 = df.loc[user2].values
            user_similarity_matrix_cosine.loc[user1, user2] = cosine_similarity_scratch(vec1, vec2)

print("\nMatrice de Similarité Cosinus (User-User) :")
print(user_similarity_matrix_cosine.round(4))

# Calculons la matrice de similarité film (item-item)
item_similarity_matrix_cosine = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

for film1 in df.columns:
    for film2 in df.columns:
        if film1 == film2:
            item_similarity_matrix_cosine.loc[
                film1, film2] = 1.0  # Un film est parfaitement similaire à lui-même
        else:
            vec1 = df[film1].values
            vec2 = df[film2].values
            item_similarity_matrix_cosine.loc[film1, film2] = cosine_similarity_scratch(vec1, vec2)

print("\nMatrice de Similarité Cosinus (Item-Item) :")
print(item_similarity_matrix_cosine.round(4))


def euclidean_similarity_scratch(vec1, vec2):
    """
    Calcule la similarité Euclidienne entre deux vecteurs NumPy.
    Gère les valeurs NaN en les ignorant.
    La similarité est calculée comme 1 / (1 + distance)
    """
    common_items_mask = ~np.isnan(vec1) & ~np.isnan(vec2)

    vec1_filtered = vec1[common_items_mask]
    vec2_filtered = vec2[common_items_mask]

    if len(vec1_filtered) == 0 or len(vec2_filtered) == 0:
        return 0.0

    distance = np.linalg.norm(vec1_filtered - vec2_filtered)

    # Convertir la distance en similarité (plus la distance est petite, plus la similarité est grande)
    return 1 / (1 + distance)


# Exemple d'utilisation de la similarité Euclidienne
print("\n--- Similarité Euclidienne ---")
# Comparons Alice et Bob
sim_euclid_alice_bob = euclidean_similarity_scratch(alice_ratings, bob_ratings)
print(f"Similarité Euclidienne entre Alice et Bob : {sim_euclid_alice_bob:.4f}")

# Comparons Charlie et David
sim_euclid_charlie_david = euclidean_similarity_scratch(charlie_ratings, david_ratings)
print(f"Similarité Euclidienne entre Charlie et David : {sim_euclid_charlie_david:.4f}")

# Calculons la matrice de similarité utilisateur (user-user) avec Euclidienne
user_similarity_matrix_euclidean = pd.DataFrame(index=df.index, columns=df.index, dtype=float)

for user1 in df.index:
    for user2 in df.index:
        if user1 == user2:
            user_similarity_matrix_euclidean.loc[user1, user2] = 1.0
        else:
            vec1 = df.loc[user1].values
            vec2 = df.loc[user2].values
            user_similarity_matrix_euclidean.loc[user1, user2] = euclidean_similarity_scratch(vec1,
                                                                                              vec2)

print("\nMatrice de Similarité Euclidienne (User-User) :")
print(user_similarity_matrix_euclidean.round(4))

# Calculons la matrice de similarité utilisateur (Item-Item) avec Euclidienne
item_similarity_matrix_euclidean = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

for film1 in df.columns:
    for film2 in df.columns:
        if film1 == film2:
            item_similarity_matrix_euclidean.loc[film1, film2] = 1.0
        else:
            vec1 = df[film1].values
            vec2 = df[film2].values
            item_similarity_matrix_euclidean.loc[film1, film2] = euclidean_similarity_scratch(vec1,
                                                                                              vec2)

print("\nMatrice de Similarité Euclidienne (Item-Item) :")
print(item_similarity_matrix_euclidean.round(4))
