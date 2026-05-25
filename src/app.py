from data_loader import MovieLensDataLoader
from preprocess import build_user_item_matrix, get_top_movies


def main() -> None:
    loader = MovieLensDataLoader()

    try:
        ratings = loader.load_ratings()
        movies = loader.load_movies()
    except FileNotFoundError:
        print("Dataset not found. Put MovieLens files in data/raw/ml-latest-small/.")
        return

    user_item_matrix = build_user_item_matrix(ratings)
    top_movies = get_top_movies(ratings)

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("User-item matrix shape:", user_item_matrix.shape)
    print("\nTop 10 popular movies:")
    print(top_movies.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
