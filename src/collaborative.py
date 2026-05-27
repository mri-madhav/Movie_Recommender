import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between users."""
    filled = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled)
    return pd.DataFrame(similarity, index=filled.index, columns=filled.index)


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between items."""
    filled = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled.T)
    return pd.DataFrame(similarity, index=filled.columns, columns=filled.columns)


def recommend_for_user(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    neighbor_count: int = 5,
) -> pd.DataFrame:
    """Recommend unseen movies for a user using user-user collaborative filtering."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in the user-item matrix.")

    filled_matrix = user_item_matrix.fillna(0)
    target_ratings = filled_matrix.loc[user_id]

    similar_users = (
        user_similarity.loc[user_id]
        .drop(labels=[user_id])
        .sort_values(ascending=False)
        .head(neighbor_count)
    )

    weighted_scores = pd.Series(0.0, index=filled_matrix.columns)
    similarity_sums = pd.Series(0.0, index=filled_matrix.columns)

    for neighbor_id, similarity_score in similar_users.items():
        neighbor_ratings = filled_matrix.loc[neighbor_id]
        rated_mask = neighbor_ratings > 0

        weighted_scores.loc[rated_mask] += neighbor_ratings.loc[rated_mask] * similarity_score
        similarity_sums.loc[rated_mask] += similarity_score

    predicted_scores = weighted_scores.div(similarity_sums.where(similarity_sums != 0, 1.0))
    unseen_mask = target_ratings == 0
    recommendations = predicted_scores.loc[unseen_mask].sort_values(ascending=False).head(top_n)

    result = recommendations.reset_index()
    result.columns = ["movieId", "predicted_rating"]
    return result.merge(movies[["movieId", "title"]], on="movieId", how="left")[
        ["movieId", "title", "predicted_rating"]
    ]


def recommend_items_for_user(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    similar_item_count: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies for a user using item-item collaborative filtering."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in the user-item matrix.")

    filled_matrix = user_item_matrix.fillna(0)
    target_ratings = filled_matrix.loc[user_id]
    rated_movies = target_ratings[target_ratings > 0]

    if rated_movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "predicted_rating"])

    scores = pd.Series(0.0, index=filled_matrix.columns)
    similarity_sums = pd.Series(0.0, index=filled_matrix.columns)

    for movie_id, rating in rated_movies.items():
        if movie_id not in item_similarity.index:
            continue

        similar_items = (
            item_similarity.loc[movie_id]
            .drop(labels=[movie_id], errors="ignore")
            .sort_values(ascending=False)
            .head(similar_item_count)
        )

        scores.loc[similar_items.index] += similar_items * rating
        similarity_sums.loc[similar_items.index] += similar_items

    predicted_scores = scores.div(similarity_sums.where(similarity_sums != 0, 1.0))
    unseen_mask = target_ratings == 0
    recommendations = predicted_scores.loc[unseen_mask].sort_values(ascending=False).head(top_n)

    result = recommendations.reset_index()
    result.columns = ["movieId", "predicted_rating"]
    return result.merge(movies[["movieId", "title"]], on="movieId", how="left")[
        ["movieId", "title", "predicted_rating"]
    ]
