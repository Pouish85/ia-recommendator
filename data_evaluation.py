import pandas as pd
import numpy as np
import random

# Dataset original (AVEC TOUTES LES NOTES INITIALES)
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

index = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Marcel']
df = pd.DataFrame(data, index=index)

print("Notre dataset initial (df original) :")
print(df)


# --- Fonctions Utilitaires ---

def calculate_user_averages(ratings_df):
    """
    Calcule la note moyenne de chaque utilisateur, en ignorant les NaN.

    :param ratings_df: DataFrame des notes (utilisateurs en index, films en colonnes)
    :return: Series Pandas avec les moyennes de notes par utilisateur
    """
    return ratings_df.mean(axis=1)


def cosine_similarity_scratch(vec1, vec2):
    """
    Calcule la similarité Cosinus entre deux vecteurs NumPy.
    Gère les valeurs NaN en les ignorant.
    """
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


# --- Fonctions de Prédiction ---

def predict_user_based(user_to_predict, movie_to_predict, ratings_df, user_similarity_matrix,
                       user_averages):
    """
    Prédit la note d'un utilisateur pour un film non noté en utilisant le filtrage collaboratif User-Based.
    """
    if user_to_predict not in user_averages:
        return np.nan

    user_mean_rating = user_averages.loc[user_to_predict]
    numerator = 0.0
    denominator = 0.0

    similarities_with_user = user_similarity_matrix.loc[user_to_predict]

    for neighbor_user in ratings_df.index:
        if neighbor_user == user_to_predict:
            continue

        neighbor_rating_for_movie = ratings_df.loc[neighbor_user, movie_to_predict]

        if not pd.isna(neighbor_rating_for_movie):
            similarity = similarities_with_user.loc[neighbor_user]
            if pd.isna(similarity) or abs(similarity) < 0.1:
                continue

            neighbor_mean_rating = user_averages.loc[neighbor_user]
            numerator += similarity * (neighbor_rating_for_movie - neighbor_mean_rating)
            denominator += abs(similarity)

    if denominator == 0:
        return np.nan

    predicted_rating = user_mean_rating + (numerator / denominator)
    predicted_rating = max(1.0, min(5.0, predicted_rating))
    return predicted_rating


def predict_item_based(user_to_predict, movie_to_predict, ratings_df, item_similarity_matrix):
    """
    Prédit la note d'un utilisateur pour un film non noté en utilisant le filtrage collaboratif Item-Based.
    """
    numerator = 0.0
    denominator = 0.0

    similarities_with_movie = item_similarity_matrix.loc[movie_to_predict]

    for rated_movie in ratings_df.columns:
        user_rating_for_rated_movie = ratings_df.loc[user_to_predict, rated_movie]

        if not pd.isna(user_rating_for_rated_movie) and rated_movie != movie_to_predict:
            similarity = similarities_with_movie.loc[rated_movie]
            if pd.isna(similarity) or abs(similarity) < 0.1:
                continue

            numerator += similarity * user_rating_for_rated_movie
            denominator += abs(similarity)

    if denominator == 0:
        return np.nan

    predicted_rating = numerator / denominator
    predicted_rating = max(1.0, min(5.0, predicted_rating))
    return predicted_rating


# --- Fonction de Recommandation (tu peux la laisser ici ou la déplacer après l'évaluation) ---

def get_recommendations(user_to_recommend, num_recommendations, ratings_df, user_similarity_matrix,
                        item_similarity_matrix, user_averages, method='user_based'):
    """
    Génère une liste de films recommandés pour un utilisateur donné.
    """
    rated_movies = ratings_df.loc[user_to_recommend].dropna().index.tolist()
    all_movies = ratings_df.columns.tolist()
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies]

    predicted_ratings = []

    for movie in unrated_movies:
        predicted_score = np.nan

        if method == 'user_based':
            predicted_score = predict_user_based(user_to_recommend, movie, ratings_df,
                                                 user_similarity_matrix, user_averages)
        elif method == 'item_based':
            predicted_score = predict_item_based(user_to_recommend, movie, ratings_df,
                                                 item_similarity_matrix)
        else:
            print(f"Méthode '{method}' non reconnue. Utilisez 'user_based' ou 'item_based'.")
            return []

        if not pd.isna(predicted_score):
            predicted_ratings.append((movie, predicted_score))

    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    return predicted_ratings[:num_recommendations]


# --- Phase 3: Évaluation du Moteur ---

print("\n--- Phase 3: Évaluation du Moteur ---")

# Créer une copie du DataFrame pour l'entraînement
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


# --- Métriques d'Évaluation ---

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


# Collecter les prédictions pour l'évaluation
user_based_predictions = []
item_based_predictions = []
user_based_actual_ratings = []  # Liste pour les notes réelles du User-Based
item_based_actual_ratings = []  # Liste pour les notes réelles de l'Item-Based

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

# ---Test avec un dataset plus grand---

# Dataset original (AVEC TOUTES LES NOTES INITIALES)
data2 = {
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

index2 = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Marcel', 'Nina', 'Oscar', 'Paul', 'Quentin', 'Rita', 'Sophie', 'Tina', 'Ursula', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zoe', 'Aaron', 'Bella', 'Cleo', 'Diana', 'Ethan', 'Fiona']
df2 = pd.DataFrame(data2, index=index2)

print("Notre dataset initial (df original) :")
print(df2)

# expected_num_users = len(index2)
# for film, ratings_list in data2.items():
#     if len(ratings_list) != expected_num_users:
#         print(f"ATTENTION : Le '{film}' a {len(ratings_list)} notes, mais il devrait en avoir {expected_num_users}.")
#     else:
#         print(f"'{film}' : OK ({len(ratings_list)} notes)")
#
# # Création du DataFrame
# df = pd.DataFrame(data2, index=index2)
# print("\nNouveau dataset (df) :")
# print(df)
# print(f"\nDimensions du nouveau dataset : {df.shape}")

print("\n--- Phase 3: Évaluation du Moteur avec dataSet plus grand ---")

# Créer une copie du DataFrame pour l'entraînement
df_train2 = df2.copy()
test_set2 = []

random.seed(42)
num_items_to_hide = 1
for user in df_train2.index:
    user_rated_movies = df_train2.loc[user].dropna().index.tolist()

    if len(user_rated_movies) > num_items_to_hide:
        movies_to_hide = random.sample(user_rated_movies, num_items_to_hide)

        for movie in movies_to_hide:
            original_rating = df_train2.loc[user, movie]
            test_set2.append({'user': user, 'movie': movie, 'original_rating': original_rating})
            df_train2.loc[user, movie] = np.nan

print(f"\nDataFrame d'entraînement (notes cachées) :\n{df_train2}")
print(f"\nEnsemble de test (notes originales cachées) :\n{pd.DataFrame(test_set2)}")

# IMPORTANT : Recalculer les matrices de similarité et les moyennes avec df_train
user_averages_train2 = calculate_user_averages(df_train2)
print("\n--- Notes moyennes par utilisateur (avec df_train2) ---")
print(user_averages_train2)

user_similarity_matrix_cosine_train2 = pd.DataFrame(index=df_train2.index, columns=df_train2.index,
                                                   dtype=float)
for user1 in df_train2.index:
    for user2 in df_train2.index:
        if user1 == user2:
            user_similarity_matrix_cosine_train2.loc[user1, user2] = 1.0
        else:
            vec1 = df_train2.loc[user1].values
            vec2 = df_train2.loc[user2].values
            user_similarity_matrix_cosine_train2.loc[user1, user2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (User-User, sur df_train2) :")
print(user_similarity_matrix_cosine_train2.round(4))

df_train_transposed2 = df_train2.T
item_similarity_matrix_cosine_train2 = pd.DataFrame(index=df_train_transposed2.index,
                                                   columns=df_train_transposed2.index, dtype=float)
for item1 in df_train_transposed2.index:
    for item2 in df_train_transposed2.index:
        if item1 == item2:
            item_similarity_matrix_cosine_train2.loc[item1, item2] = 1.0
        else:
            vec1 = df_train_transposed2.loc[item1].values
            vec2 = df_train_transposed2.loc[item2].values
            item_similarity_matrix_cosine_train2.loc[item1, item2] = cosine_similarity_scratch(vec1,
                                                                                              vec2)
print("\nMatrice de Similarité Cosinus (Item-Item, sur df_train) :")
print(item_similarity_matrix_cosine_train2.round(4))

# Collecter les prédictions pour l'évaluation
user_based_predictions2 = []
item_based_predictions2 = []
user_based_actual_ratings2 = []  # Liste pour les notes réelles du User-Based
item_based_actual_ratings2 = []  # Liste pour les notes réelles de l'Item-Based

for entry in test_set2:
    user = entry['user']
    movie = entry['movie']
    original_rating = entry['original_rating']

    # Prédiction User-Based
    predicted_user_based2 = predict_user_based(user, movie, df_train2,
                                              user_similarity_matrix_cosine_train2,
                                              user_averages_train2)
    if not pd.isna(predicted_user_based2):
        user_based_predictions2.append(predicted_user_based2)
        user_based_actual_ratings2.append(original_rating)

    # Prédiction Item-Based
    predicted_item_based = predict_item_based(user, movie, df_train2,
                                              item_similarity_matrix_cosine_train2)
    if not pd.isna(predicted_item_based):
        item_based_predictions2.append(predicted_item_based)
        item_based_actual_ratings2.append(original_rating)

print("\n--- Résultats d'Évaluation ---")

# Évaluation User-Based
if user_based_predictions2:
    mae_ub, rmse_ub = evaluate_recommender(user_based_predictions2, user_based_actual_ratings2)
    print(f"User-Based Recommender (N={len(user_based_predictions2)} prédictions) :")
    print(f"  MAE : {mae_ub:.4f}")
    print(f"  RMSE : {rmse_ub:.4f}")
else:
    print("Pas assez de prédictions User-Based pour l'évaluation.")

# Évaluation Item-Based
if item_based_predictions2:
    mae_ib, rmse_ib = evaluate_recommender(item_based_predictions2, item_based_actual_ratings2)
    print(f"Item-Based Recommender (N={len(item_based_predictions2)} prédictions) :")
    print(f"  MAE : {mae_ib:.4f}")
    print(f"  RMSE : {rmse_ib:.4f}")
else:
    print("Pas assez de prédictions Item-Based pour l'évaluation.")
