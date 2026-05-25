import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_genre_similarity(movies: pd.DataFrame) -> pd.DataFrame:
    """Create a simple content-based similarity matrix using genres."""
    genre_text = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(genre_text)
    similarity = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(similarity, index=movies["movieId"], columns=movies["movieId"])
