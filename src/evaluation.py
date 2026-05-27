import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.collaborative import (
    compute_item_similarity,
    compute_user_similarity,
    recommend_for_user,
    recommend_items_for_user,
)
from src.content_based import build_content_similarity, recommend_content_for_user
from src.matrix_factorization import reconstruct_ratings, recommend_from_predicted_matrix
from src.preprocess import get_top_movies


def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Compute mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def precision_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Precision@K for one user."""
    if k <= 0:
        return 0.0
    hits = len(set(recommended_ids[:k]) & relevant_ids)
    return hits / k


def hit_rate_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Hit Rate@K for one user."""
    return float(len(set(recommended_ids[:k]) & relevant_ids) > 0)


def evaluate_recommenders(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None = None,
    k: int = 10,
    min_relevant_rating: float = 4.0,
    max_users: int | None = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate recommenders using ranking metrics on a holdout test set."""
    user_item_matrix = train_ratings.pivot_table(index="userId", columns="movieId", values="rating")
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    content_similarity = build_content_similarity(movies, tags)
    predicted_matrix = reconstruct_ratings(user_item_matrix, n_components=20)
    popularity = get_top_movies(train_ratings, min_ratings=20)

    train_seen_by_user = train_ratings.groupby("userId")["movieId"].apply(set).to_dict()
    relevant_by_user = (
        test_ratings[test_ratings["rating"] >= min_relevant_rating]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    candidate_users = list(relevant_by_user.keys())
    if max_users is not None and len(candidate_users) > max_users:
        rng = np.random.default_rng(random_state)
        sampled_users = rng.choice(candidate_users, size=max_users, replace=False)
        relevant_by_user = {int(user_id): relevant_by_user[int(user_id)] for user_id in sampled_users}

    metric_store = {
        "popularity": {"precision": [], "hit_rate": []},
        "user_user_cf": {"precision": [], "hit_rate": []},
        "item_item_cf": {"precision": [], "hit_rate": []},
        "content_based": {"precision": [], "hit_rate": []},
        "matrix_factorization": {"precision": [], "hit_rate": []},
    }

    for user_id, relevant_ids in relevant_by_user.items():
        if user_id not in user_item_matrix.index:
            continue

        seen_ids = train_seen_by_user.get(user_id, set())
        if not seen_ids:
            continue

        popularity_ids = [
            movie_id for movie_id in popularity["movieId"].tolist() if movie_id not in seen_ids
        ][:k]
        user_user_ids = recommend_for_user(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity=user_similarity,
            movies=movies,
            top_n=k,
            neighbor_count=5,
        )["movieId"].tolist()
        item_item_ids = recommend_items_for_user(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            item_similarity=item_similarity,
            movies=movies,
            top_n=k,
            similar_item_count=10,
        )["movieId"].tolist()
        content_ids = recommend_content_for_user(
            user_id=user_id,
            ratings=train_ratings,
            movies=movies,
            content_similarity=content_similarity,
            top_n=k,
            min_rating=min_relevant_rating,
        )["movieId"].tolist()
        matrix_ids = recommend_from_predicted_matrix(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            predicted_matrix=predicted_matrix,
            movies=movies,
            top_n=k,
        )["movieId"].tolist()

        model_predictions = {
            "popularity": popularity_ids,
            "user_user_cf": user_user_ids,
            "item_item_cf": item_item_ids,
            "content_based": content_ids,
            "matrix_factorization": matrix_ids,
        }

        for model_name, recommended_ids in model_predictions.items():
            metric_store[model_name]["precision"].append(precision_at_k(recommended_ids, relevant_ids, k))
            metric_store[model_name]["hit_rate"].append(hit_rate_at_k(recommended_ids, relevant_ids, k))

    rows = []
    for model_name, metrics in metric_store.items():
        user_count = len(metrics["precision"])
        rows.append(
            {
                "model": model_name,
                "evaluated_users": user_count,
                f"precision@{k}": float(np.mean(metrics["precision"])) if user_count else 0.0,
                f"hit_rate@{k}": float(np.mean(metrics["hit_rate"])) if user_count else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(by=f"precision@{k}", ascending=False).reset_index(drop=True)
