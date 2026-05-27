# Movie Recommendation System

This project is a modular recommendation system built. It is designed to compare multiple recommendation approaches on the MovieLens dataset:

- User-user collaborative filtering
- Item-item collaborative filtering
- Content-based filtering
- Matrix factorization

The first dataset used is MovieLens because it contains both user ratings and movie metadata, which makes it a strong benchmark for recommendation system projects.

## What Has Been Implemented

So far, the project includes:

- A clean project structure with separate modules for loading, preprocessing, recommendation logic, and evaluation
- A single source-of-truth data loader in `src/data_loader.py`
- A preprocessing pipeline that builds a user-item matrix from MovieLens ratings
- A popularity-based baseline recommender
- A user-user collaborative filtering recommender using cosine similarity
- An item-item collaborative filtering recommender using cosine similarity between movies
- A content-based recommender using genres, title metadata, and optional user tags
- A main application entry point that prints baseline, user-user, item-item, and content-based recommendations for a sample user

## Current Recommendation Flow

The current version of the system works like this:

1. Load `ratings.csv` and `movies.csv`
2. Build a user-item matrix where rows are users and columns are movies
3. Compute cosine similarity between users, between movies, and between movie metadata vectors
4. For user-user filtering, find the nearest users to a target user
5. Predict scores for unseen movies using weighted ratings from similar users
6. For item-item filtering, score unseen movies based on similarity to movies the user already rated
7. For content-based filtering, recommend movies similar to the metadata of movies the user rated highly
8. Return top movie recommendations

## Interview Explanation

You can explain the current implementation like this:

> I structured the project in a modular way so data loading, preprocessing, recommendation algorithms, and evaluation are separated. I first implemented a popularity baseline, then built user-user and item-item collaborative filtering models using a user-item rating matrix and cosine similarity. I also added a content-based recommender that uses genres, title text, and tags to represent each movie, then recommends movies similar to the ones a user has already rated highly.

## Project Structure

```text
Movie_Recommendation_System/
  data/
    raw/
    processed/
  notebooks/
  src/
    __init__.py
    app.py
    collaborative.py
    content_based.py
    data_loader.py
    evaluation.py
    matrix_factorization.py
    preprocess.py
  DATASET_GUIDE.md
  QUICK_DOWNLOAD_GUIDE.md
  setup_dataset.py
  requirements.txt
  README.md
```

## File Roles

- `src/data_loader.py`: loads MovieLens ratings and movie metadata
- `src/preprocess.py`: creates the user-item matrix and popularity baseline
- `src/collaborative.py`: contains collaborative filtering logic
- `src/content_based.py`: builds metadata-based movie similarity and content-based recommendations
- `src/matrix_factorization.py`: contains matrix factorization logic
- `src/evaluation.py`: contains evaluation metrics such as RMSE and MAE
- `src/app.py`: main script that ties the pipeline together
- `setup_dataset.py`: helper script for dataset download and setup

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset and place it in:

```text
data/raw/ml-latest-small/
```

4. Run the application:

```bash
python -m src.app
```

## Next Steps

The next planned improvements are:

- Add matrix factorization comparison
- Add evaluation metrics across all models
- Optionally build a small Streamlit interface
