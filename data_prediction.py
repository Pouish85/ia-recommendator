import pandas as pd
import numpy as np

# Dataset simple : Utilisateurs (lignes) vs. Films/Produits (colonnes)
# Les valeurs représentent les notes (par exemple, de 1 à 5)
data = {
    'Film A': [5, 4, np.nan, 1, 3, 4],
    'Film B': [3, np.nan, 4, 2, 5, 4],
    'Film C': [4, 5, 2, np.nan, 4, 4],
    'Film D': [np.nan, 3, 5, 4, 2, np.nan],
    'Film E': [2, 4, np.nan, 3, np.nan, 3],
    'Film F': [np.nan, 2, np.nan, 5, 4, np.nan],
    'Film G': [4, np.nan, 3, np.nan, 5, 4],
    'Film H': [np.nan, 5, np.nan, 3, np.nan, 2]
}

# Créez le DataFrame Pandas avec des noms d'utilisateurs comme index
index = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Marcel']
df = pd.DataFrame(data, index=index)

print("Notre dataset initial :")
print(df)

def calculate_user_averages(ratings_df):
    """
    Calcule la note moyenne de chaque utilisateur, en ignorant les NaN.

    :param ratings_df: DataFrame des notes (utilisateurs en index, films en colonnes)
    :return: Series Pandas avec les moyennes de notes par utilisateur
    """
    # axis=1 signifie que l'on calcule la moyenne sur les lignes (par utilisateur)
    # skipna=True est le comportement par défaut pour .mean(), il ignore les NaN
    return ratings_df.mean(axis=1)

# Calcule les moyennes de notes pour tous les utilisateurs
user_averages = calculate_user_averages(df)
print("\n--- Notes moyennes par utilisateur ---")
print(user_averages)

def cosine_similarity_scratch(vec1, vec2):
    """
    Calcule la similarité Cosinus entre deux vecteurs NumPy.
    Gère les valeurs NaN en les ignorant.
    """
    # Masque pour les valeurs non NaN (présentes dans les deux vecteurs)
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

user_similarity_matrix_cosine = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
for user1 in df.index:
    for user2 in df.index:
        if user1 == user2:
            user_similarity_matrix_cosine.loc[user1, user2] = 1.0
        else:
            vec1 = df.loc[user1].values
            vec2 = df.loc[user2].values
            user_similarity_matrix_cosine.loc[user1, user2] = cosine_similarity_scratch(vec1, vec2)

print("\nMatrice de Similarité Cosinus (User-User) :")
print(user_similarity_matrix_cosine.round(4))


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


# --- Exemples d'utilisation de la prédiction User-Based ---
print("\n--- Prédictions User-Based ---")

# Prédire la note d'Alice pour Film D (qu'elle n'a pas noté)
alice_film_d_pred = predict_user_based('Alice', 'Film D', df, user_similarity_matrix_cosine,
                                       user_averages)
print(f"Note prédite pour Alice sur Film D : {alice_film_d_pred:.4f}")

# Prédire la note de Charlie pour Film A (qu'il n'a pas noté)
charlie_film_a_pred = predict_user_based('Charlie', 'Film A', df, user_similarity_matrix_cosine,
                                         user_averages)
print(f"Note prédite pour Charlie sur Film A : {charlie_film_a_pred:.4f}")

# Prédire la note de Bob pour Film B (qu'il n'a pas noté)
bob_film_b_pred = predict_user_based('Bob', 'Film B', df, user_similarity_matrix_cosine,
                                     user_averages)
print(f"Note prédite pour Bob sur Film B : {bob_film_b_pred:.4f}")

# Prédire la note de David pour Film C (qu'il n'a pas noté)
david_film_c_pred = predict_user_based('David', 'Film C', df, user_similarity_matrix_cosine,
                                       user_averages)
print(f"Note prédite pour David sur Film C : {david_film_c_pred:.4f}")

# Prédire la note d'Eve pour Film D (qu'elle a noté - juste pour tester la fonction, le NaN sera ignoré)
eve_film_d_pred = predict_user_based('Eve', 'Film D', df, user_similarity_matrix_cosine,
                                     user_averages)
print(
    f"Note prédite pour Eve sur Film D : {eve_film_d_pred:.4f}")  # Elle a noté 2.0, la prédiction devrait être proche.

# Prédire la note de Marcel pour Film D (qu'il n'a pas noté)
marcel_film_d_pred = predict_user_based('Marcel', 'Film D', df, user_similarity_matrix_cosine,
                                     user_averages)
print(
    f"Note prédite pour Marcel sur Film D : {marcel_film_d_pred:.4f}")

# Prédire la note de Alice pour Film H (qu'elle n'a pas noté)
alice_film_h_pred = predict_user_based('Alice', 'Film H', df, user_similarity_matrix_cosine,
                                     user_averages)
print(
    f"Note prédite pour Alice sur Film H : {alice_film_h_pred:.4f}")

# Transposer le DataFrame pour avoir les films en index et les utilisateurs en colonnes
# C'est nécessaire pour calculer la similarité entre les films
df_transposed = df.T

# Initialiser la matrice de similarité Item-Item
item_similarity_matrix_cosine = pd.DataFrame(index=df_transposed.index, columns=df_transposed.index, dtype=float)

for item1 in df_transposed.index:
    for item2 in df_transposed.index:
        if item1 == item2:
            item_similarity_matrix_cosine.loc[item1, item2] = 1.0
        else:
            vec1 = df_transposed.loc[item1].values
            vec2 = df_transposed.loc[item2].values
            item_similarity_matrix_cosine.loc[item1, item2] = cosine_similarity_scratch(vec1, vec2)

print("\nMatrice de Similarité Cosinus (Item-Item) :")
print(item_similarity_matrix_cosine.round(4))


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


# --- Exemples d'utilisation de la prédiction Item-Based ---
print("\n--- Prédictions Item-Based ---")

# Prédire la note d'Alice pour Film D (qu'elle n'a pas noté)
alice_film_d_pred_item = predict_item_based('Alice', 'Film D', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour Alice sur Film D : {alice_film_d_pred_item:.4f}")

# Prédire la note de Charlie pour Film A (qu'il n'a pas noté)
charlie_film_a_pred_item = predict_item_based('Charlie', 'Film A', df,
                                              item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour Charlie sur Film A : {charlie_film_a_pred_item:.4f}")

# Prédire la note de Bob pour Film B (qu'il n'a pas noté)
bob_film_b_pred_item = predict_item_based('Bob', 'Film B', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour Bob sur Film B : {bob_film_b_pred_item:.4f}")

# Prédire la note de David pour Film C (qu'il n'a pas noté)
david_film_c_pred_item = predict_item_based('David', 'Film C', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour David sur Film C : {david_film_c_pred_item:.4f}")

# Prédire la note de Marcel pour Film D (qu'il n'a pas noté)
marcel_film_d_pred_item = predict_item_based('Marcel', 'Film D', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour Marcel sur Film D : {marcel_film_d_pred_item:.4f}")

# Prédire la note de Eve pour Film H (qu'il n'a pas noté)
eve_film_h_pred_item = predict_item_based('Eve', 'Film H', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour Eve sur Film H : {eve_film_h_pred_item:.4f}")

# Prédire la note de David pour Film G (qu'il n'a pas noté)
david_film_g_pred_item = predict_item_based('David', 'Film G', df, item_similarity_matrix_cosine)
print(f"Note prédite (Item-Based) pour David sur Film G : {david_film_g_pred_item:.4f}")

def get_recommendations(user_to_recommend, num_recommendations, ratings_df, user_similarity_matrix, item_similarity_matrix, user_averages, method='user_based'):
    """
        Génère une liste de films recommandés pour un utilisateur donné.

        :param user_to_recommend: Nom de l'utilisateur pour lequel générer des recommandations.
        :param num_recommendations: Nombre de films à recommander.
        :param ratings_df: DataFrame complet des notes.
        :param user_similarity_matrix: Matrice de similarité User-User (nécessaire si method='user_based').
        :param item_similarity_matrix: Matrice de similarité Item-Item (nécessaire si method='item_based').
        :param user_averages: Series des notes moyennes par utilisateur (nécessaire si method='user_based').
        :param method: Méthode de recommandation à utiliser ('user_based' ou 'item_based').
        :return: Une liste de tuples (nom_film, note_prédite) des films recommandés, triée par note.
        """
    # Identifier les films déjà notés par l'utilisateur
    # On regarde les films où la note n'est PAS NaN
    rated_movies = ratings_df.loc[user_to_recommend].dropna().index.tolist()

    # Identifier tous les films disponibles
    all_movies = ratings_df.columns.tolist()

    # Identifier les films non encore notés par l'utilisateur
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies]

    predicted_ratings = []

    for movie in unrated_movies:
        predicted_score = np.nan  # Initialiser avec NaN

        if method == 'user_based':
            # Utiliser la fonction de prédiction User-Based
            predicted_score = predict_user_based(user_to_recommend, movie, ratings_df, user_similarity_matrix,
                                                 user_averages)
        elif method == 'item_based':
            # Utiliser la fonction de prédiction Item-Based
            predicted_score = predict_item_based(user_to_recommend, movie, ratings_df, item_similarity_matrix)
        else:
            print(f"Méthode '{method}' non reconnue. Utilisez 'user_based' ou 'item_based'.")
            return []  # Retourne une liste vide en cas de méthode invalide

        # N'ajouter que les prédictions valides
        if not pd.isna(predicted_score):
            predicted_ratings.append((movie, predicted_score))

    # Trier les films par ordre décroissant de notes prédites
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Retourner les N meilleures recommandations
    return predicted_ratings[:num_recommendations]


# --- Exemples d'utilisation de la fonction get_recommendations ---
print("\n--- Recommandations Générées ---")

# Recommandations User-Based pour Alice (qui a noté A, B, C)
print("\nRecommandations User-Based pour Alice :")
recommendations_alice_user = get_recommendations(
    'Alice', 3, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='user_based'
)
if recommendations_alice_user:
    for movie, score in recommendations_alice_user:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour Alice avec la méthode User-Based.")

# Recommandations Item-Based pour Charlie (qui a noté C, D, G)
print("\nRecommandations Item-Based pour Charlie :")
recommendations_charlie_item = get_recommendations(
    'Charlie', 3, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='item_based'
)
if recommendations_charlie_item:
    for movie, score in recommendations_charlie_item:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour Charlie avec la méthode Item-Based.")

# Recommandations User-Based pour Marcel (qui a noté A, B, C, G)
print("\nRecommandations User-Based pour Marcel :")
recommendations_marcel_user = get_recommendations(
    'Marcel', 2, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='user_based'
)
if recommendations_marcel_user:
    for movie, score in recommendations_marcel_user:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour Marcel avec la méthode User-Based.")

# Recommandations Item-Based pour David (qui a noté A, B, D, F, H)
print("\nRecommandations Item-Based pour David :")
recommendations_david_item = get_recommendations(
    'David', 2, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='item_based'
)
if recommendations_david_item:
    for movie, score in recommendations_david_item:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour David avec la méthode Item-Based.")

# Recommandations Item-Based pour Bob (qui a noté A, C, D, E, F, H)
print("\nRecommandations Item-Based pour Bob :")
recommendations_bob_item = get_recommendations(
    'Bob', 2, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='item_based'
)
if recommendations_bob_item:
    for movie, score in recommendations_bob_item:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour Bob avec la méthode Item-Based.")

# Recommandations Item-Based pour Eve (qui a noté A, B, C, D, F, G)
print("\nRecommandations Item-Based pour Eve :")
recommendations_eve_item = get_recommendations(
    'Eve', 2, df, user_similarity_matrix_cosine, item_similarity_matrix_cosine, user_averages,
    method='item_based'
)
if recommendations_eve_item:
    for movie, score in recommendations_eve_item:
        print(f"- {movie}: {score:.4f}")
else:
    print("Aucune recommandation trouvée pour Eve avec la méthode Item-Based.")

