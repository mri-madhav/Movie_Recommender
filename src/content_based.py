import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_movie_metadata(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine genres, title tokens, and optional tags into one metadata text field."""
    movies_metadata = movies.copy()
    movies_metadata["genres_text"] = movies_metadata["genres"].fillna("").str.replace("|", " ", regex=False)
    movies_metadata["title_text"] = (
        movies_metadata["title"]
        .fillna("")
        .str.replace(r"\(\d{4}\)", "", regex=True)
        .str.replace(r"[^A-Za-z0-9 ]", " ", regex=True)
    )

    if tags is not None and not tags.empty:
        tag_summary = (
            tags.groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.astype(str)))
            .reset_index(name="tag_text")
        )
        movies_metadata = movies_metadata.merge(tag_summary, on="movieId", how="left")
    else:
        movies_metadata["tag_text"] = ""

    movies_metadata["metadata_text"] = (
        movies_metadata["genres_text"].fillna("")
        + " "
        + movies_metadata["title_text"].fillna("")
        + " "
        + movies_metadata["tag_text"].fillna("")
    ).str.strip()

    return movies_metadata


def build_content_similarity(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a content-based similarity matrix using genres, title text, and optional tags."""
    movies_metadata = build_movie_metadata(movies, tags)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_metadata["metadata_text"])
    similarity = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(
        similarity,
        index=movies_metadata["movieId"],
        columns=movies_metadata["movieId"],
    )


def recommend_content_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_similarity: pd.DataFrame,
    top_n: int = 10,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Recommend unseen movies using metadata similarity to the user's liked movies."""
    user_ratings = ratings[ratings["userId"] == user_id]
    liked_movies = user_ratings[user_ratings["rating"] >= min_rating][["movieId", "rating"]]

    if liked_movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "content_score"])

    seen_movie_ids = set(user_ratings["movieId"].tolist())
    scores = pd.Series(0.0, index=content_similarity.columns)
    similarity_sums = pd.Series(0.0, index=content_similarity.columns)

    for movie_id, rating in liked_movies.itertuples(index=False):
        if movie_id not in content_similarity.index:
            continue

        similar_movies = content_similarity.loc[movie_id].drop(labels=[movie_id], errors="ignore")
        scores.loc[similar_movies.index] += similar_movies * rating
        similarity_sums.loc[similar_movies.index] += similar_movies

    content_scores = scores.div(similarity_sums.where(similarity_sums != 0, 1.0))
    unseen_scores = content_scores.drop(labels=list(seen_movie_ids), errors="ignore")
    recommendations = unseen_scores.sort_values(ascending=False).head(top_n)

    result = recommendations.reset_index()
    result.columns = ["movieId", "content_score"]
    return result.merge(movies[["movieId", "title"]], on="movieId", how="left")[
        ["movieId", "title", "content_score"]
    ]
