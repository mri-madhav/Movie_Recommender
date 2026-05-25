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
