import pandas as pd


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item ratings matrix."""
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
    )


def get_top_movies(ratings: pd.DataFrame, min_ratings: int = 50) -> pd.DataFrame:
    """Return popular movies by average rating and rating count."""
    summary = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
        .reset_index()
    )
    return summary[summary["rating_count"] >= min_ratings].sort_values(
        by=["avg_rating", "rating_count"],
        ascending=[False, False],
    )
