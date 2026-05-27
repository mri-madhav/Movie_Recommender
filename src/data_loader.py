from pathlib import Path

import pandas as pd


class MovieLensDataLoader:
    """Loads MovieLens ratings and movie metadata from the local data directory."""

    def __init__(self, dataset_dir: str = "data/raw/ml-latest-small") -> None:
        self.dataset_dir = Path(dataset_dir)

    def load_ratings(self) -> pd.DataFrame:
        ratings_path = self.dataset_dir / "ratings.csv"
        return pd.read_csv(ratings_path)

    def load_movies(self) -> pd.DataFrame:
        movies_path = self.dataset_dir / "movies.csv"
        return pd.read_csv(movies_path)

    def load_tags(self) -> pd.DataFrame:
        tags_path = self.dataset_dir / "tags.csv"
        return pd.read_csv(tags_path)

    def load_merged(self) -> pd.DataFrame:
        ratings = self.load_ratings()
        movies = self.load_movies()
        return ratings.merge(movies, on="movieId", how="left")
