from src.data_loader import MovieLensDataLoader
from src.preprocess import attach_movie_titles, build_user_item_matrix, get_top_movies
from src.collaborative import (
    compute_item_similarity,
    compute_user_similarity,
    recommend_for_user,
    recommend_items_for_user,
)
from src.content_based import build_content_similarity, recommend_content_for_user
from src.evaluation import evaluate_recommenders
from src.matrix_factorization import recommend_matrix_factorization
from sklearn.model_selection import train_test_split


def main() -> None:
    loader = MovieLensDataLoader()

    try:
        ratings = loader.load_ratings()
        movies = loader.load_movies()
    except FileNotFoundError:
        print("Dataset not found. Put MovieLens files in data/raw/ml-latest-small/.")
        return

    try:
        tags = loader.load_tags()
    except FileNotFoundError:
        tags = None

    user_item_matrix = build_user_item_matrix(ratings)
    top_movies = attach_movie_titles(get_top_movies(ratings), movies)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    content_similarity = build_content_similarity(movies, tags)
    sample_user_id = int(ratings["userId"].iloc[0])
    user_user_recommendations = recommend_for_user(
        user_id=sample_user_id,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies=movies,
        top_n=10,
        neighbor_count=5,
    )
    item_item_recommendations = recommend_items_for_user(
        user_id=sample_user_id,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies=movies,
        top_n=10,
        similar_item_count=10,
    )
    content_recommendations = recommend_content_for_user(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies,
        content_similarity=content_similarity,
        top_n=10,
        min_rating=4.0,
    )
    matrix_factorization_recommendations = recommend_matrix_factorization(
        user_id=sample_user_id,
        user_item_matrix=user_item_matrix,
        movies=movies,
        top_n=10,
        n_components=20,
    )
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    evaluation_results = evaluate_recommenders(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        movies=movies,
        tags=tags,
        k=10,
        min_relevant_rating=4.0,
        max_users=150,
        random_state=42,
    )

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("User-item matrix shape:", user_item_matrix.shape)
    print("\nTop 10 popular movies:")
    print(top_movies[["movieId", "title", "avg_rating", "rating_count"]].head(10).to_string(index=False))
    print(f"\nTop 10 user-user recommendations for user {sample_user_id}:")
    print(user_user_recommendations.to_string(index=False))
    print(f"\nTop 10 item-item recommendations for user {sample_user_id}:")
    print(item_item_recommendations.to_string(index=False))
    print(f"\nTop 10 content-based recommendations for user {sample_user_id}:")
    print(content_recommendations.to_string(index=False))
    print(f"\nTop 10 matrix factorization recommendations for user {sample_user_id}:")
    print(matrix_factorization_recommendations.to_string(index=False))
    print("\nEvaluation summary (holdout split):")
    print(evaluation_results.to_string(index=False))


if __name__ == "__main__":
    main()
