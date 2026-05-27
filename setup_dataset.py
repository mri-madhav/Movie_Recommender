#!/usr/bin/env python3
"""
Movie Recommendation System - Dataset Setup Script
Downloads and prepares the MovieLens dataset for this project structure.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


class DatasetDownloader:
    def __init__(self, output_dir: str = "./data/raw") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_movielens(self, size: str = "small") -> Path | None:
        """Download and extract a MovieLens dataset."""
        if size == "small":
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            zip_name = "ml-latest-small.zip"
            extract_dir = "ml-latest-small"
        elif size == "100k":
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            zip_name = "ml-100k.zip"
            extract_dir = "ml-100k"
        elif size == "1m":
            url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            zip_name = "ml-1m.zip"
            extract_dir = "ml-1m"
        else:
            raise ValueError(f"Size {size} not supported. Choose: small, 100k, 1m")

        zip_path = self.output_dir / zip_name
        extract_path = self.output_dir / extract_dir

        if extract_path.exists():
            print(f"Dataset already exists at {extract_path}")
            return extract_path

        print(f"Downloading MovieLens {size} dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded to {zip_path}")
        except Exception as exc:
            print(f"Download failed: {exc}")
            return None

        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
            print(f"Extracted to {extract_path}")
        except Exception as exc:
            print(f"Extraction failed: {exc}")
            return None

        return extract_path

    def load_and_verify(self, dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None] | tuple[None, None, None]:
        """Load the downloaded data and print quick checks."""
        print(f"\nLoading dataset from {dataset_path}...")

        try:
            ratings = pd.read_csv(dataset_path / "ratings.csv")
            movies = pd.read_csv(dataset_path / "movies.csv")

            tags_path = dataset_path / "tags.csv"
            tags = pd.read_csv(tags_path) if tags_path.exists() else None

            print("\n=== DATASET OVERVIEW ===")
            print(f"Ratings loaded: {len(ratings):,} entries")
            print(f"Movies loaded: {len(movies):,} entries")
            if tags is not None:
                print(f"Tags loaded: {len(tags):,} entries")

            print("\n=== STATISTICS ===")
            print(f"Unique users: {ratings['userId'].nunique():,}")
            print(f"Unique movies: {ratings['movieId'].nunique():,}")
            print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
            print(f"Average rating: {ratings['rating'].mean():.2f}")

            total_possible = ratings["userId"].nunique() * ratings["movieId"].nunique()
            sparsity = 1 - len(ratings) / total_possible
            print(f"Matrix sparsity: {sparsity:.2%}")

            return ratings, movies, tags
        except Exception as exc:
            print(f"Error loading dataset: {exc}")
            return None, None, None

    def save_processed_data(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
        output_name: str = "processed_data",
    ) -> tuple[Path, Path]:
        """Save merged data to the raw data area for easier reuse."""
        print("\nSaving processed data...")
        merged = ratings.merge(movies, on="movieId")

        pickle_path = self.output_dir / f"{output_name}.pkl"
        csv_path = self.output_dir / f"{output_name}.csv"

        merged.to_pickle(pickle_path)
        merged.to_csv(csv_path, index=False)

        print(f"Saved merged data to {pickle_path}")
        print(f"Saved merged data to {csv_path}")

        return pickle_path, csv_path

    def create_train_test_split(
        self,
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create a simple train/test split for later evaluation."""
        from sklearn.model_selection import train_test_split

        print(f"\nCreating train/test split ({100 * (1 - test_size):.0f}/{100 * test_size:.0f})...")
        train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)

        print(f"Training set: {len(train):,} ratings")
        print(f"Test set: {len(test):,} ratings")

        return train, test


def main() -> None:
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - DATASET SETUP")
    print("=" * 60)

    downloader = DatasetDownloader("./data/raw")

    print("\n[1/4] Downloading MovieLens dataset...")
    dataset_path = downloader.download_movielens("small")
    if dataset_path is None:
        print("Failed to download dataset")
        sys.exit(1)

    print("\n[2/4] Loading and verifying data...")
    ratings, movies, tags = downloader.load_and_verify(dataset_path)
    if ratings is None or movies is None:
        print("Failed to load dataset")
        sys.exit(1)

    print("\n[3/4] Processing and saving data...")
    downloader.save_processed_data(ratings, movies, tags)

    print("\n[4/4] Creating train/test split...")
    train, test = downloader.create_train_test_split(ratings)

    processed_dir = Path("./data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train_ratings.csv", index=False)
    test.to_csv(processed_dir / "test_ratings.csv", index=False)
    print("Saved train/test splits to ./data/processed/")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nYour data is ready at: ./data/")
    print("  - raw/ml-latest-small/         (Original dataset)")
    print("  - raw/processed_data.pkl       (Merged data)")
    print("  - raw/processed_data.csv       (Merged data)")
    print("  - processed/train_ratings.csv  (Training set)")
    print("  - processed/test_ratings.csv   (Test set)")


if __name__ == "__main__":
    main()
