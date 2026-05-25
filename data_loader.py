"""
Recommendation System - Dataset & Utility Module
Ready-to-use classes for loading, exploring, and preprocessing recommendation data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class MovieLensLoader:
    """Load and process MovieLens datasets"""
    
    def __init__(self, dataset_dir='./ml-latest-small'):
        self.dataset_dir = Path(dataset_dir)
        self.ratings = None
        self.movies = None
        self.tags = None
        self.merged_data = None
    
    def load(self):
        """Load all MovieLens files"""
        print(f"📂 Loading dataset from {self.dataset_dir}...")
        
        self.ratings = pd.read_csv(self.dataset_dir / 'ratings.csv')
        self.movies = pd.read_csv(self.dataset_dir / 'movies.csv')
        
        # Load tags if available
        tags_file = self.dataset_dir / 'tags.csv'
        if tags_file.exists():
            self.tags = pd.read_csv(tags_file)
        
        # Merge ratings with movie info
        self.merged_data = self.ratings.merge(self.movies, on='movieId')
        
        print(f"✓ Loaded {len(self.ratings):,} ratings")
        print(f"✓ Loaded {len(self.movies):,} movies")
        if self.tags is not None:
            print(f"✓ Loaded {len(self.tags):,} tags")
        
        return self
    
    def get_statistics(self):
        """Print dataset statistics"""
        if self.ratings is None:
            print("❌ Load data first using .load()")
            return
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        stats = {
            'Total Ratings': len(self.ratings),
            'Unique Users': self.ratings['userId'].nunique(),
            'Unique Movies': self.ratings['movieId'].nunique(),
            'Average Rating': f"{self.ratings['rating'].mean():.2f}",
            'Rating Std Dev': f"{self.ratings['rating'].std():.2f}",
            'Min Rating': self.ratings['rating'].min(),
            'Max Rating': self.ratings['rating'].max(),
        }
        
        # Matrix sparsity
        total_possible = self.ratings['userId'].nunique() * self.ratings['movieId'].nunique()
        sparsity = 1 - len(self.ratings) / total_possible
        stats['Matrix Sparsity'] = f"{sparsity:.2%}"
        
        # Activity
        stats['Avg Ratings per User'] = f"{len(self.ratings) / self.ratings['userId'].nunique():.1f}"
        stats['Avg Ratings per Movie'] = f"{len(self.ratings) / self.ratings['movieId'].nunique():.1f}"
        
        # Most rated
        most_rated_id = self.ratings['movieId'].value_counts().index[0]
        most_rated_title = self.movies[self.movies['movieId'] == most_rated_id]['title'].values[0]
        stats['Most Rated Movie'] = f"{most_rated_title} ({self.ratings['movieId'].value_counts().iloc[0]} ratings)"
        
        # Most active user
        most_active_id = self.ratings['userId'].value_counts().index[0]
        most_active_count = self.ratings['userId'].value_counts().iloc[0]
        stats['Most Active User'] = f"User {most_active_id} ({most_active_count} ratings)"
        
        for key, value in stats.items():
            print(f"{key:<25}: {value}")
        
        print("="*50 + "\n")
        
        return stats
    
    def get_sample_data(self, n=5):
        """Display sample data"""
        print(f"\n📊 SAMPLE RATINGS (first {n}):")
        print(self.ratings.head(n).to_string())
        
        print(f"\n🎬 SAMPLE MOVIES (first {n}):")
        print(self.movies.head(n).to_string())
        
        print(f"\n📝 SAMPLE MERGED DATA (first {n}):")
        print(self.merged_data.head(n).to_string())
    
    def create_utility_matrix(self, fillna=0):
        """Create user-item rating matrix"""
        print("\n🔄 Creating utility matrix...")
        
        matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=fillna
        )
        
        print(f"✓ Matrix shape: {matrix.shape}")
        print(f"✓ Sparsity: {(matrix == 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]):.2%}")
        
        return matrix
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data for train/test evaluation"""
        print(f"\n🔀 Creating train/test split ({100*(1-test_size):.0f}/{100*test_size:.0f})...")
        
        train, test = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"✓ Training set: {len(train):,} ratings")
        print(f"✓ Test set: {len(test):,} ratings")
        
        return train, test
    
    def get_user_items(self, user_id):
        """Get all items rated by a user"""
        user_ratings = self.merged_data[self.merged_data['userId'] == user_id]
        return user_ratings.sort_values('rating', ascending=False)
    
    def get_movie_info(self, movie_id):
        """Get info for a specific movie"""
        movie = self.movies[self.movies['movieId'] == movie_id]
        if movie.empty:
            print(f"❌ Movie {movie_id} not found")
            return None
        return movie.iloc[0]
    
    def filter_data(self, min_user_ratings=0, min_movie_ratings=0):
        """Filter users/movies with minimum activity"""
        print(f"\n🔍 Filtering data...")
        
        original_size = len(self.ratings)
        
        if min_user_ratings > 0:
            active_users = self.ratings['userId'].value_counts()
            active_users = active_users[active_users >= min_user_ratings].index
            self.ratings = self.ratings[self.ratings['userId'].isin(active_users)]
            print(f"✓ Filtered to users with >= {min_user_ratings} ratings")
        
        if min_movie_ratings > 0:
            active_movies = self.ratings['movieId'].value_counts()
            active_movies = active_movies[active_movies >= min_movie_ratings].index
            self.ratings = self.ratings[self.ratings['movieId'].isin(active_movies)]
            print(f"✓ Filtered to movies with >= {min_movie_ratings} ratings")
        
        print(f"  Ratings: {original_size:,} → {len(self.ratings):,}")
        
        # Recreate merged data
        self.merged_data = self.ratings.merge(self.movies, on='movieId')
        
        return self
    
    def get_genre_stats(self):
        """Analyze genre distribution"""
        print("\n🎬 GENRE STATISTICS:")
        
        # Split genres and expand
        genre_list = self.movies['genres'].str.split('|').explode()
        genre_counts = genre_list.value_counts()
        
        print(f"Total unique genres: {len(genre_counts)}")
        print("\nTop 10 genres:")
        for genre, count in genre_counts.head(10).items():
            print(f"  {genre:<15}: {count:>5} movies")
        
        return genre_counts
    
    def save(self, filename='processed_data.pkl'):
        """Save processed data"""
        import pickle
        
        data = {
            'ratings': self.ratings,
            'movies': self.movies,
            'tags': self.tags,
            'merged_data': self.merged_data
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Saved data to {filename}")
    
    def load_saved(self, filename='processed_data.pkl'):
        """Load previously saved data"""
        import pickle
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.ratings = data['ratings']
        self.movies = data['movies']
        self.tags = data['tags']
        self.merged_data = data['merged_data']
        
        print(f"✓ Loaded data from {filename}")
        return self


class SpotifyLoader:
    """Load Spotify dataset"""
    
    def __init__(self, filepath='spotify_data.csv'):
        self.filepath = filepath
        self.data = None
    
    def load(self):
        """Load Spotify data"""
        self.data = pd.read_csv(self.filepath)
        print(f"✓ Loaded {len(self.data):,} songs")
        return self
    
    def get_statistics(self):
        """Print statistics"""
        if self.data is None:
            print("❌ Load data first")
            return
        
        print("\n" + "="*50)
        print("SPOTIFY DATASET STATISTICS")
        print("="*50)
        
        print(f"Total songs: {len(self.data):,}")
        print(f"Unique artists: {self.data['artist'].nunique() if 'artist' in self.data.columns else 'N/A'}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Audio features if available
        audio_cols = [col for col in self.data.columns if col in [
            'danceability', 'energy', 'tempo', 'valence', 'acousticness'
        ]]
        
        if audio_cols:
            print(f"\nAudio Features:")
            print(self.data[audio_cols].describe())
        
        print("="*50)


class RecommendationEvaluator:
    """Evaluate recommendation models"""
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def precision_at_k(recommendations, test_data, k=10):
        """Precision@K metric"""
        precisions = []
        
        for user_id, user_items in test_data.groupby('userId'):
            user_test_items = set(user_items['movieId'].values)
            
            if user_id in recommendations:
                user_recs = recommendations[user_id][:k]
                hits = len(set(user_recs) & user_test_items)
                precision = hits / min(k, len(user_test_items))
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0
    
    @staticmethod
    def coverage(recommendations, total_items):
        """Fraction of items that have recommendations"""
        unique_items = set()
        for recs in recommendations.values():
            unique_items.update(recs)
        
        return len(unique_items) / total_items


# Example usage
if __name__ == '__main__':
    print("🎬 MOVIE RECOMMENDATION SYSTEM - DATA LOADER DEMO\n")
    
    # Load MovieLens data
    loader = MovieLensLoader('./ml-latest-small')
    loader.load()
    
    # Get statistics
    loader.get_statistics()
    
    # Display samples
    loader.get_sample_data(3)
    
    # Create utility matrix
    utility_matrix = loader.create_utility_matrix()
    print(f"\nUtility Matrix (10x10 sample):")
    print(utility_matrix.iloc[:10, :10])
    
    # Create train/test split
    train, test = loader.train_test_split(test_size=0.2)
    
    # Genre stats
    loader.get_genre_stats()
    
    # Filter for active users
    loader_filtered = MovieLensLoader('./ml-latest-small')
    loader_filtered.load().filter_data(min_user_ratings=20, min_movie_ratings=5)
    
    print("\n✓ Data loader is ready to use!")
