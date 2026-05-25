# 🎬 Movie/Music Recommendation System - Dataset Guide

## 📊 Dataset Options

### **OPTION 1: MovieLens (RECOMMENDED - Easiest) ✅**

**Best for**: Quick start, classic ML showcase, widely recognized

#### Available Versions:
| Version | Size | Ratings | Users | Movies | Best For |
|---------|------|---------|-------|--------|----------|
| **ml-latest-small** | 10 MB | 100K | 600 | 9K | **Hackathon (quickest)** ⭐ |
| ml-100k | 5 MB | 100K | 943 | 1.6K | Learning/benchmark |
| ml-1m | 28 MB | 1M | 6K | 3.9K | Medium-scale |
| ml-10m | 250 MB | 10M | 72K | 10.8K | Large-scale |
| ml-20m | 1.2 GB | 20M | 138K | 26.8K | Research |
| ml-latest-full | 1.2 GB | 33M | 330K | 86K | **Large-scale research** |

#### Download Links:

```bash
# Option A: Direct Download (ml-latest-small - FASTEST)
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cd ml-latest-small

# Option B: ml-100k (Classic, smaller)
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip

# Option C: ml-1m (Medium size, good balance)
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

#### File Structure:
```
ml-latest-small/
├── ratings.csv      # userId, movieId, rating, timestamp
├── movies.csv       # movieId, title, genres
├── tags.csv         # userId, movieId, tag, timestamp
└── links.csv        # movieId, imdbId, tmdbId
```

#### Python Code to Load & Explore:

```python
import pandas as pd
import numpy as np

# Load MovieLens data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# Quick exploration
print("=== Dataset Overview ===")
print(f"Total ratings: {len(ratings):,}")
print(f"Unique users: {ratings['userId'].nunique():,}")
print(f"Unique movies: {ratings['movieId'].nunique():,}")
print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
print(f"Sparsity: {1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.2%}")

print("\n=== Sample Ratings ===")
print(ratings.head())

print("\n=== Sample Movies ===")
print(movies.head())

# Merge data
data = ratings.merge(movies, on='movieId')
print("\n=== Merged Data ===")
print(data.head())

# Data statistics
print(f"\nAverage rating: {ratings['rating'].mean():.2f}")
print(f"Most rated movie: {movies.iloc[ratings['movieId'].value_counts().idxmax()]['title']}")
print(f"Most active user rated: {ratings['userId'].value_counts().max()} movies")
```

---

### **OPTION 2: Spotify Dataset**

**Best for**: Music recommendation, audio features, modern use case

#### Available on Kaggle:

1. **Spotify Tracks Dataset**
   - URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
   - Size: ~500K songs
   - Features: audio features (danceability, energy, tempo, etc.)

2. **Million Song Dataset + Spotify + Last.fm**
   - URL: https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm
   - Size: 1M+ songs
   - Features: song metadata + listening patterns

3. **Spotify Music Dataset**
   - URL: https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset
   - Size: Popular and unpopular songs
   - Features: popularity, genres, artist info

#### How to Download from Kaggle:

```bash
# Step 1: Install Kaggle CLI
pip install kaggle

# Step 2: Get API credentials
# - Go to https://www.kaggle.com/settings/account
# - Click "Create New API Token"
# - Save kaggle.json to ~/.kaggle/

# Step 3: Download dataset
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip spotify-tracks-dataset.zip
```

#### Load Spotify Data:

```python
import pandas as pd

# Load Spotify tracks
spotify = pd.read_csv('spotify_data.csv')

print("=== Spotify Dataset Overview ===")
print(f"Total songs: {len(spotify):,}")
print(f"Columns: {spotify.columns.tolist()}")
print(f"Genres: {spotify['genre'].nunique()}")

print("\n=== Audio Features ===")
print(spotify[['danceability', 'energy', 'tempo', 'valence']].describe())

print("\n=== Sample Data ===")
print(spotify.head())
```

---

### **OPTION 3: Last.fm Dataset**

**Best for**: Real user listening patterns, implicit feedback

#### Sources:

1. **Last.fm Dataset from Kaggle**
   - URL: https://www.kaggle.com/datasets/harshal19t/lastfm-dataset
   - Format: CSV with user-artist-play counts

2. **Last.fm Dataset 2020 (GitHub)**
   - URL: https://github.com/renesemela/lastfm-dataset-2020
   - Format: SQLite database
   - Features: 122K tracks, 100 tags, Spotify preview links

#### Download & Load:

```python
import pandas as pd

# If using CSV version
lastfm = pd.read_csv('lastfm_data.csv')

print("=== Last.fm Dataset Overview ===")
print(f"Total plays: {len(lastfm):,}")
print(f"Unique users: {lastfm['user'].nunique():,}")
print(f"Unique artists: {lastfm['artist'].nunique():,}")

# If using SQLite version
import sqlite3
conn = sqlite3.connect('lastfm_dataset_2020.db')
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# Query data
metadata = pd.read_sql_query("SELECT * FROM metadata LIMIT 10", conn)
print(metadata.head())
```

---

### **OPTION 4: IMDb Dataset**

**Best for**: Movie metadata enrichment, genre-based recommendations

#### Sources:

1. **IMDb Datasets (Official)**
   - URL: https://developer.imdb.com/non-commercial-datasets/
   - Format: TSV files
   - Data: Ratings, genres, cast, directors

#### Download IMDb:

```bash
# Download IMDb datasets
wget https://datasets.imdbws.com/name.basics.tsv.gz
wget https://datasets.imdbws.com/title.ratings.tsv.gz
wget https://datasets.imdbws.com/title.basics.tsv.gz

# Extract
gunzip *.gz

# Load in Python
import pandas as pd

imdb_ratings = pd.read_csv('title.ratings.tsv', sep='\t')
imdb_titles = pd.read_csv('title.basics.tsv', sep='\t')

print(imdb_ratings.head())
print(imdb_titles.head())
```

---

## 🚀 Quick Start Setup

### Install Required Libraries:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

### Create a Simple Data Loader:

```python
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, dataset_type='movielens'):
        self.dataset_type = dataset_type
    
    def load_movielens(self, size='small'):
        """Load MovieLens dataset"""
        base_path = f'ml-latest-{size}'
        
        ratings = pd.read_csv(f'{base_path}/ratings.csv')
        movies = pd.read_csv(f'{base_path}/movies.csv')
        
        data = ratings.merge(movies, on='movieId')
        return data, ratings, movies
    
    def get_statistics(self, ratings):
        """Print dataset statistics"""
        stats = {
            'total_ratings': len(ratings),
            'unique_users': ratings['userId'].nunique(),
            'unique_items': ratings['movieId'].nunique(),
            'avg_rating': ratings['rating'].mean(),
            'sparsity': 1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique())
        }
        
        for key, value in stats.items():
            print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")
        
        return stats

# Usage
loader = DataLoader()
data, ratings, movies = loader.load_movielens('small')
loader.get_statistics(ratings)
```

---

## 📈 Dataset Comparison for Your Project

| Dataset | Size | Speed | Features | Best For |
|---------|------|-------|----------|----------|
| **MovieLens Small** | 10 MB | ⚡⚡⚡ | Ratings only | **Hackathon (Go This!)** |
| **MovieLens 1M** | 28 MB | ⚡⚡ | Ratings + metadata | Good balance |
| **Spotify** | 500+ MB | ⚡ | Audio features | Music recommendations |
| **Last.fm** | 1+ GB | 🔶 | Listening history | Advanced features |

---

## 🎯 Recommendation for Your Hackathon:

### **Use MovieLens ml-latest-small** ✅

**Why?**
- ✅ Quick download (10 MB)
- ✅ Fast to load and process
- ✅ Complete ratings data
- ✅ Movie metadata included
- ✅ Can showcase 3 different algorithms easily
- ✅ Widely recognized in ML community

### **Step-by-Step Setup (5 minutes):**

```bash
# 1. Download
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# 2. Extract
unzip ml-latest-small.zip

# 3. Navigate
cd ml-latest-small

# 4. Create Python script to verify
python3 << 'EOF'
import pandas as pd
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
print(f"✓ Loaded {len(ratings):,} ratings")
print(f"✓ Loaded {len(movies):,} movies")
print(f"✓ Ready to build recommendation systems!")
EOF
```

---

## 📚 Next Steps:

1. **Download** the dataset using command above
2. **Load & explore** with the Python code provided
3. **Start building** collaborative filtering, content-based, and matrix factorization models
4. **Evaluate** with RMSE, MAE, precision@k metrics
5. **Deploy** as Flask API for demo

---

## 🔗 Useful Resources:

- **GroupLens (Official)**: https://grouplens.org/datasets/movielens/
- **Kaggle Datasets**: https://www.kaggle.com/datasets?search=movie+recommendation
- **Million Song Dataset**: http://millionsongdataset.com/
- **Recommendation Systems Tutorial**: https://d2l.ai/chapter_recommender-systems/

---

**Happy Hacking! 🚀**
