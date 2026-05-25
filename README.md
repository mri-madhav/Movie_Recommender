# Movie Recommendation System

This project is a starter scaffold for a recommendation system that compares:

- User-user collaborative filtering
- Item-item collaborative filtering
- Content-based filtering
- Matrix factorization

The recommended first dataset is MovieLens because it includes user ratings and movie metadata.

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
  data_loader.py
  setup_dataset.py
  requirements.txt
  README.md
```

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Download MovieLens into `data/raw/ml-latest-small/`.
4. Start by running the loader and preprocessing modules.

## First Build Goal

The first milestone is to:

- Load MovieLens ratings and movies data
- Build a user-item matrix
- Create a simple popularity baseline
- Add user-user and item-item collaborative filtering

## Notes

- Existing root-level helper files were kept as-is.
- New work should go into the `src/` folder.
