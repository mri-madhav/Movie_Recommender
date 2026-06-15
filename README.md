# Movie Recommendation System

A movie recommendation system built with the MovieLens dataset. It compares several recommendation methods and includes a Streamlit demo.

## Methods

- Popularity-based recommendation
- User-user collaborative filtering
- Item-item collaborative filtering
- Content-based filtering
- Matrix factorization with TruncatedSVD

## Project Structure

```text
Movie_Recommendation_System/
  data/
    raw/
    processed/
  notebooks/
  src/
    app.py
    collaborative.py
    content_based.py
    data_loader.py
    evaluation.py
    matrix_factorization.py
    preprocess.py
    streamlit_app.py
  requirements.txt
  setup_dataset.py
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python setup_dataset.py
```

Run the command-line app:

```bash
python -m src.app
```

Run the Streamlit app:

```bash
python -m streamlit run src/streamlit_app.py
```

## Evaluation

The models are compared using a holdout split with:

- Precision@10
- Hit Rate@10
