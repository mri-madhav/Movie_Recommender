import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def fit_svd(user_item_matrix: pd.DataFrame, n_components: int = 20) -> tuple[TruncatedSVD, np.ndarray]:
    """Fit a basic matrix factorization model using TruncatedSVD."""
    filled = user_item_matrix.fillna(0)
    safe_components = max(1, min(n_components, filled.shape[0] - 1, filled.shape[1] - 1))
    model = TruncatedSVD(n_components=safe_components, random_state=42)
    latent_factors = model.fit_transform(filled)
    return model, latent_factors


def reconstruct_ratings(
    user_item_matrix: pd.DataFrame,
    n_components: int = 20,
) -> pd.DataFrame:
    """Reconstruct approximate user-item ratings from latent factors."""
    model, user_factors = fit_svd(user_item_matrix, n_components=n_components)
    reconstructed = np.dot(user_factors, model.components_)
    return pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )


def recommend_from_predicted_matrix(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    predicted_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies from a precomputed predicted ratings matrix."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in the user-item matrix.")

    seen_mask = user_item_matrix.loc[user_id].notna()
    predicted_scores = predicted_matrix.loc[user_id].loc[~seen_mask].sort_values(ascending=False).head(top_n)

    result = predicted_scores.reset_index()
    result.columns = ["movieId", "predicted_rating"]
    return result.merge(movies[["movieId", "title"]], on="movieId", how="left")[
        ["movieId", "title", "predicted_rating"]
    ]


def recommend_matrix_factorization(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    n_components: int = 20,
) -> pd.DataFrame:
    """Recommend unseen movies using matrix factorization with TruncatedSVD."""
    predicted_matrix = reconstruct_ratings(user_item_matrix, n_components=n_components)
    return recommend_from_predicted_matrix(user_id, user_item_matrix, predicted_matrix, movies, top_n=top_n)
