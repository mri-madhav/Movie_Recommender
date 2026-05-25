#!/usr/bin/env python3
"""
Movie/Music Recommendation System - Dataset Setup Script
Automatically downloads and sets up MovieLens dataset for your hackathon project
"""

import os
import urllib.request
import zipfile
import pandas as pd
import sys
from pathlib import Path

class DatasetDownloader:
    def __init__(self, output_dir='./data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_movielens(self, size='small'):
        """
        Download MovieLens dataset
        
        Args:
            size: 'small' (100K), '1m', '10m', '20m'
        """
        
        if size == 'small':
            url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
            zip_name = 'ml-latest-small.zip'
            extract_dir = 'ml-latest-small'
        elif size == '100k':
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            zip_name = 'ml-100k.zip'
            extract_dir = 'ml-100k'
        elif size == '1m':
            url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
            zip_name = 'ml-1m.zip'
            extract_dir = 'ml-1m'
        else:
            raise ValueError(f"Size {size} not supported. Choose: small, 100k, 1m")
        
        zip_path = self.output_dir / zip_name
        extract_path = self.output_dir / extract_dir
        
        # Skip if already exists
        if extract_path.exists():
            print(f"✓ Dataset already exists at {extract_path}")
            return extract_path
        
        # Download
        print(f"📥 Downloading MovieLens {size} dataset ({url})...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"✓ Downloaded to {zip_path}")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
        
        # Extract
        print(f"📦 Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            print(f"✓ Extracted to {extract_path}")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return None
        
        return extract_path
    
    def load_and_verify(self, dataset_path):
        """Load and verify dataset integrity"""
        
        print(f"\n📊 Loading dataset from {dataset_path}...")
        
        try:
            # Load CSV files
            ratings = pd.read_csv(f'{dataset_path}/ratings.csv')
            movies = pd.read_csv(f'{dataset_path}/movies.csv')
            
            # Load tags if available
            tags_path = f'{dataset_path}/tags.csv'
            tags = None
            if os.path.exists(tags_path):
                tags = pd.read_csv(tags_path)
            
            print("\n=== DATASET OVERVIEW ===")
            print(f"✓ Ratings loaded: {len(ratings):,} entries")
            print(f"✓ Movies loaded: {len(movies):,} entries")
            if tags is not None:
                print(f"✓ Tags loaded: {len(tags):,} entries")
            
            print("\n=== STATISTICS ===")
            print(f"Unique users: {ratings['userId'].nunique():,}")
            print(f"Unique movies: {ratings['movieId'].nunique():,}")
            print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
            print(f"Average rating: {ratings['rating'].mean():.2f}")
            
            # Matrix sparsity
            total_possible = ratings['userId'].nunique() * ratings['movieId'].nunique()
            sparsity = 1 - len(ratings) / total_possible
            print(f"Matrix sparsity: {sparsity:.2%}")
            
            print("\n=== SAMPLE DATA ===")
            print("\nRatings (first 5):")
            print(ratings.head())
            
            print("\nMovies (first 5):")
            print(movies.head())
            
            print("\n=== GENRES ===")
            print(f"Unique genres: {movies['genres'].nunique()}")
            
            return ratings, movies, tags
            
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None, None, None
    
    def save_processed_data(self, ratings, movies, tags=None, output_name='processed_data'):
        """Save processed data for easier access"""
        
        print(f"\n💾 Saving processed data...")
        
        # Merge ratings with movie info
        merged = ratings.merge(movies, on='movieId')
        
        # Save as pickle (faster loading)
        pickle_path = self.output_dir / f'{output_name}.pkl'
        merged.to_pickle(pickle_path)
        print(f"✓ Saved merged data to {pickle_path}")
        
        # Save as CSV
        csv_path = self.output_dir / f'{output_name}.csv'
        merged.to_csv(csv_path, index=False)
        print(f"✓ Saved merged data to {csv_path}")
        
        return pickle_path, csv_path
    
    def create_train_test_split(self, ratings, test_size=0.2, random_state=42):
        """Create train/test split for evaluation"""
        
        print(f"\n🔀 Creating train/test split ({100*(1-test_size):.0f}/{100*test_size:.0f})...")
        
        from sklearn.model_selection import train_test_split
        
        train, test = train_test_split(
            ratings, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"✓ Training set: {len(train):,} ratings")
        print(f"✓ Test set: {len(test):,} ratings")
        
        return train, test

def main():
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM - DATASET SETUP 🎬")
    print("=" * 60)
    
    # Create downloader
    downloader = DatasetDownloader('./data')
    
    # Download dataset
    print("\n[1/4] Downloading MovieLens dataset...")
    dataset_path = downloader.download_movielens('small')
    
    if dataset_path is None:
        print("✗ Failed to download dataset")
        sys.exit(1)
    
    # Load and verify
    print("\n[2/4] Loading and verifying data...")
    ratings, movies, tags = downloader.load_and_verify(dataset_path)
    
    if ratings is None:
        print("✗ Failed to load dataset")
        sys.exit(1)
    
    # Save processed data
    print("\n[3/4] Processing and saving data...")
    downloader.save_processed_data(ratings, movies, tags)
    
    # Create train/test split
    print("\n[4/4] Creating train/test split...")
    train, test = downloader.create_train_test_split(ratings)
    
    # Save splits
    train.to_csv('./data/train_ratings.csv', index=False)
    test.to_csv('./data/test_ratings.csv', index=False)
    print(f"✓ Saved train/test splits to ./data/")
    
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE!")
    print("=" * 60)
    print("\n📁 Your data is ready at: ./data/")
    print("   - ml-latest-small/     (Original dataset)")
    print("   - processed_data.pkl   (Merged data)")
    print("   - processed_data.csv   (Merged data)")
    print("   - train_ratings.csv    (Training set)")
    print("   - test_ratings.csv     (Test set)")
    print("\n🚀 Next steps:")
    print("   1. Build collaborative filtering model")
    print("   2. Build content-based filtering model")
    print("   3. Implement matrix factorization")
    print("   4. Evaluate on test set")
    print("   5. Create Flask API for deployment")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
