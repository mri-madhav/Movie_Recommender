import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def fit_svd(user_item_matrix: pd.DataFrame, n_components: int = 20) -> tuple[TruncatedSVD, np.ndarray]:
    """Fit a basic matrix factorization model using TruncatedSVD."""
    filled = user_item_matrix.fillna(0)
    model = TruncatedSVD(n_components=n_components, random_state=42)
    latent_factors = model.fit_transform(filled)
    return model, latent_factors
