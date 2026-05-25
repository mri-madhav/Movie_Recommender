# 🎯 Quick Reference - Download Commands

## ⚡ FASTEST SETUP (MovieLens - Recommended)

```bash
# Step 1: Download (takes ~30 seconds)
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# Step 2: Extract
unzip ml-latest-small.zip

# Step 3: Done! Navigate to dataset
cd ml-latest-small
ls -la  # You'll see: ratings.csv, movies.csv, tags.csv, links.csv
```

### What you get:
- `ratings.csv`: 100,000 ratings from 600 users on 9,000 movies
- `movies.csv`: Movie titles and genres
- `tags.csv`: User-applied tags to movies
- File size: ~10 MB total

---

## 🎬 ALL MOVIELENS VERSIONS

### Quick Comparison:
```
ml-latest-small  →  100K ratings    (10 MB)   ⭐ BEST FOR HACKATHON
ml-100k          →  100K ratings    (5 MB)    Classic baseline
ml-1m            →  1M ratings      (28 MB)   Medium project
ml-10m           →  10M ratings     (250 MB)  Large-scale
ml-20m           →  20M ratings     (1.2 GB)  Research-grade
```

### Download Commands:

```bash
# Small (100K ratings) - FASTEST ⭐
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# 100K (Classic)
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip

# 1M
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip

# 10M (Requires more resources)
curl -O https://files.grouplens.org/datasets/movielens/ml-10m.zip

# 20M (Research-grade, very large)
curl -O https://files.grouplens.org/datasets/movielens/ml-20m.zip

# Full latest (33M+ ratings)
curl -O https://files.grouplens.org/datasets/movielens/ml-latest.zip
```

---

## 🎵 SPOTIFY DATASETS

### Download via Kaggle CLI:

```bash
# Install Kaggle CLI
pip install kaggle

# Configure (go to https://www.kaggle.com/settings/account to get API key)
mkdir -p ~/.kaggle
# Paste your kaggle.json here

# Option 1: Spotify Tracks Dataset
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip spotify-tracks-dataset.zip

# Option 2: Spotify Music Dataset
kaggle datasets download -d solomonameh/spotify-music-dataset
unzip spotify-music-dataset.zip

# Option 3: Million Song Dataset + Spotify + Last.fm
kaggle datasets download -d undefinenull/million-song-dataset-spotify-lastfm
unzip million-song-dataset-spotify-lastfm.zip
```

### Direct Browser Download (No CLI):
- https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset

---

## 🎙️ LAST.FM DATASETS

### Option 1: Kaggle CSV
```bash
kaggle datasets download -d harshal19t/lastfm-dataset
unzip lastfm-dataset.zip
```
**URL**: https://www.kaggle.com/datasets/harshal19t/lastfm-dataset

### Option 2: GitHub SQLite Database
```bash
git clone https://github.com/renesemela/lastfm-dataset-2020.git
cd lastfm-dataset-2020
# Database file: lastfm_dataset_2020.db
```
**URL**: https://github.com/renesemela/lastfm-dataset-2020

### Option 3: Million Song Dataset + Last.fm
```
URL: http://millionsongdataset.com/lastfm/
Size: Large dataset with JSON files
Format: Raw data in JSON format
```

---

## 📽️ IMDB DATASETS

```bash
# Download IMDb datasets
wget https://datasets.imdbws.com/name.basics.tsv.gz
wget https://datasets.imdbws.com/title.ratings.tsv.gz
wget https://datasets.imdbws.com/title.basics.tsv.gz

# Extract
gunzip *.gz

# Load in Python
python3 << 'EOF'
import pandas as pd
ratings = pd.read_csv('title.ratings.tsv', sep='\t')
titles = pd.read_csv('title.basics.tsv', sep='\t')
print(ratings.head())
EOF
```

**Official URL**: https://developer.imdb.com/non-commercial-datasets/

---

## 💻 PYTHON QUICK START

### Load MovieLens:
```python
import pandas as pd

# Load the three main files
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# Print basic info
print(f"Ratings: {len(ratings)}")
print(f"Users: {ratings['userId'].nunique()}")
print(f"Movies: {ratings['movieId'].nunique()}")

# Merge ratings with movie info
data = ratings.merge(movies, on='movieId')
print(data.head())
```

### Load Spotify:
```python
import pandas as pd

spotify = pd.read_csv('spotify_data.csv')
print(spotify.head())
print(f"Total songs: {len(spotify)}")
print(f"Audio features: {spotify.columns.tolist()}")
```

### Load Last.fm:
```python
import sqlite3
import pandas as pd

# SQLite version
conn = sqlite3.connect('lastfm_dataset_2020.db')
metadata = pd.read_sql_query("SELECT * FROM metadata LIMIT 10", conn)
print(metadata.head())
```

---

## 🎯 WHICH DATASET TO CHOOSE?

### For Hackathon (48 hours):
```
MovieLens ml-latest-small ⭐⭐⭐
├─ Why: Fast download, complete data, easy to use
├─ Size: 10 MB
├─ Time: 5 minutes setup
└─ Best for: Collaborative filtering + content-based + matrix factorization
```

### For Music Focus:
```
Spotify Tracks Dataset ⭐⭐
├─ Why: Audio features, genre data, modern approach
├─ Size: 500+ MB
└─ Best for: Music recommendation showcase
```

### For Implicit Feedback:
```
Last.fm Dataset ⭐⭐
├─ Why: Real listening patterns, large scale
├─ Size: 1+ GB
└─ Best for: Advanced collaborative filtering
```

### For Academic Research:
```
MovieLens ml-20m or ml-latest ⭐⭐⭐
├─ Why: Large scale, established benchmark
├─ Size: 1.2 GB+
└─ Best for: Deep learning, neural collaborative filtering
```

---

## 📊 DATASET STATISTICS AT A GLANCE

| Dataset | Users | Items | Ratings | Sparsity | Size |
|---------|-------|-------|---------|----------|------|
| **MovieLens Small** | 600 | 9K | 100K | 98.2% | 10 MB |
| MovieLens 100K | 943 | 1.6K | 100K | 93.7% | 5 MB |
| MovieLens 1M | 6K | 3.9K | 1M | 95.9% | 28 MB |
| MovieLens 10M | 72K | 10.8K | 10M | 98.6% | 250 MB |
| MovieLens 20M | 138K | 26.8K | 20M | 99.4% | 1.2 GB |
| Spotify Tracks | - | 500K+ | - | - | 500 MB |
| Last.fm | Millions | Millions | Millions | - | 1+ GB |

---

## 🛠️ AUTOMATED SETUP SCRIPT

```bash
# Download and run the automated setup script
python3 setup_dataset.py
```

This will:
1. Download MovieLens dataset
2. Load and verify data
3. Create train/test splits
4. Save processed files for easy loading

---

## 🚨 TROUBLESHOOTING

### Download fails with "Cannot reach grouplens.org"
```bash
# Try wget instead of curl
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# Or use Python
python3 -c "import urllib.request; urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-latest-small.zip', 'ml-latest-small.zip')"
```

### Kaggle authentication fails
```bash
# Re-setup Kaggle credentials
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset --force
```

### File permissions issue
```bash
chmod +x setup_dataset.py
python3 setup_dataset.py
```

### Out of memory (if using large dataset)
```bash
# Use smaller version
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# Or load in chunks
import pandas as pd
for chunk in pd.read_csv('ratings.csv', chunksize=10000):
    # Process chunk
    pass
```

---

## ✅ VERIFICATION CHECKLIST

After download, verify your setup:

```bash
# 1. Check directory structure
ls -la ml-latest-small/
# Expected output: ratings.csv, movies.csv, tags.csv, links.csv

# 2. Check file sizes
du -h ml-latest-small/
# Expected: ~10 MB total

# 3. Quick Python test
python3 << 'EOF'
import pandas as pd
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(f"✓ Loaded {len(ratings)} ratings")
print(f"✓ Shape: {ratings.shape}")
print(f"✓ Columns: {list(ratings.columns)}")
EOF
```

---

## 🎬 NEXT STEPS AFTER DOWNLOAD

1. **Load & Explore** data (descriptive statistics)
2. **Preprocess** (handle missing values, normalize ratings)
3. **Split** into train/test sets
4. **Build Models**:
   - Collaborative Filtering (User-User, Item-Item)
   - Content-Based Filtering
   - Matrix Factorization (SVD, NMF)
5. **Evaluate** using RMSE, MAE, Precision@K
6. **Visualize** recommendations
7. **Deploy** as Flask API

---

**Happy building! 🚀**
