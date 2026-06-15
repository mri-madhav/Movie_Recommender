from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# Ensure project root is importable when Streamlit sets sys.path to the script directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collaborative import (  # noqa: E402
    compute_item_similarity,
    compute_user_similarity,
    recommend_for_user,
    recommend_items_for_user,
)
from src.content_based import build_content_similarity, recommend_content_for_user  # noqa: E402
from src.data_loader import MovieLensDataLoader  # noqa: E402
from src.evaluation import evaluate_recommenders  # noqa: E402
from src.matrix_factorization import recommend_matrix_factorization  # noqa: E402
from src.preprocess import attach_movie_titles, build_user_item_matrix, get_top_movies  # noqa: E402


st.set_page_config(page_title="Movie Recommendation System", layout="wide")

METHOD_DESCRIPTIONS = {
    "Popularity": "Popular high-rated movies.",
    "User-User CF": "Recommendations from similar users.",
    "Item-Item CF": "Movies similar to what the user rated.",
    "Content-Based": "Movies matched by genres, titles, and tags.",
    "Matrix Factorization": "Recommendations from latent rating patterns.",
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    loader = MovieLensDataLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()

    try:
        tags = loader.load_tags()
    except FileNotFoundError:
        tags = None

    return ratings, movies, tags


@st.cache_data
def prepare_evaluation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None,
) -> pd.DataFrame:
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    return evaluate_recommenders(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        movies=movies,
        tags=tags,
        k=10,
        min_relevant_rating=4.0,
        max_users=150,
        random_state=42,
    )


@st.cache_resource
def build_artifacts(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    user_item_matrix = build_user_item_matrix(ratings)
    top_movies = attach_movie_titles(get_top_movies(ratings), movies)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    content_similarity = build_content_similarity(movies, tags)

    return {
        "user_item_matrix": user_item_matrix,
        "top_movies": top_movies,
        "user_similarity": user_similarity,
        "item_similarity": item_similarity,
        "content_similarity": content_similarity,
    }


def get_user_history(ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int) -> pd.DataFrame:
    history = ratings[ratings["userId"] == user_id].merge(movies, on="movieId", how="left")
    return history.sort_values(by=["rating", "timestamp"], ascending=[False, False])[
        ["movieId", "title", "genres", "rating"]
    ]


def _prepare_method_frames(
    artifacts: dict[str, pd.DataFrame],
    user_user_recommendations: pd.DataFrame,
    item_item_recommendations: pd.DataFrame,
    content_recommendations: pd.DataFrame,
    matrix_factorization_recommendations: pd.DataFrame,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    method_frames = {
        "Popularity": artifacts["top_movies"][["movieId", "title", "avg_rating", "rating_count"]].head(top_n).copy(),
        "User-User CF": user_user_recommendations.copy(),
        "Item-Item CF": item_item_recommendations.copy(),
        "Content-Based": content_recommendations.copy(),
        "Matrix Factorization": matrix_factorization_recommendations.copy(),
    }
    return method_frames


def _display_recommendation_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No recommendations available for this method and user configuration.")
        return

    for idx, row in frame.head(10).iterrows():
        title = row.get("title", "Unknown title")
        movie_id = row.get("movieId", "-")
        details = [f"Movie ID: {movie_id}"]
        if "predicted_rating" in frame.columns:
            details.append(f"Predicted rating: {row['predicted_rating']:.3f}")
        if "content_score" in frame.columns:
            details.append(f"Content score: {row['content_score']:.3f}")
        if "avg_rating" in frame.columns:
            details.append(f"Avg rating: {row['avg_rating']:.3f}")
        if "rating_count" in frame.columns:
            details.append(f"Rating count: {int(row['rating_count'])}")
        st.markdown(f"**{idx + 1}. {title}**  \n" + " | ".join(details))


def main() -> None:
    st.title("Movie Recommendation System")
    st.caption("Compare recommendation methods on MovieLens.")

    try:
        ratings, movies, tags = load_data()
    except FileNotFoundError:
        st.error("Dataset not found. Put MovieLens files in data/raw/ml-latest-small/ or run python setup_dataset.py.")
        return

    artifacts = build_artifacts(ratings, movies, tags)

    st.sidebar.header("Controls")
    user_ids = sorted(ratings["userId"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select user ID", user_ids, index=0)
    top_n = st.sidebar.slider("Recommendations per method", min_value=5, max_value=20, value=10, step=1)
    neighbor_count = st.sidebar.slider("User-user neighbors", min_value=3, max_value=20, value=5, step=1)
    similar_item_count = st.sidebar.slider("Item-item neighbors", min_value=5, max_value=30, value=10, step=1)
    min_rating = st.sidebar.slider("Minimum liked rating for content-based", min_value=3.0, max_value=5.0, value=4.0, step=0.5)
    show_evaluation = st.sidebar.checkbox("Show evaluation table", value=True)
    ui_mode = st.sidebar.radio("Recommendation view", ["Table", "Cards"], index=0)

    user_history = get_user_history(ratings, movies, selected_user)
    user_user_recommendations = recommend_for_user(
        user_id=selected_user,
        user_item_matrix=artifacts["user_item_matrix"],
        user_similarity=artifacts["user_similarity"],
        movies=movies,
        top_n=top_n,
        neighbor_count=neighbor_count,
    )
    item_item_recommendations = recommend_items_for_user(
        user_id=selected_user,
        user_item_matrix=artifacts["user_item_matrix"],
        item_similarity=artifacts["item_similarity"],
        movies=movies,
        top_n=top_n,
        similar_item_count=similar_item_count,
    )
    content_recommendations = recommend_content_for_user(
        user_id=selected_user,
        ratings=ratings,
        movies=movies,
        content_similarity=artifacts["content_similarity"],
        top_n=top_n,
        min_rating=min_rating,
    )
    matrix_factorization_recommendations = recommend_matrix_factorization(
        user_id=selected_user,
        user_item_matrix=artifacts["user_item_matrix"],
        movies=movies,
        top_n=top_n,
        n_components=20,
    )
    method_frames = _prepare_method_frames(
        artifacts=artifacts,
        user_user_recommendations=user_user_recommendations,
        item_item_recommendations=item_item_recommendations,
        content_recommendations=content_recommendations,
        matrix_factorization_recommendations=matrix_factorization_recommendations,
        top_n=top_n,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Users", f"{ratings['userId'].nunique():,}")
    metric_2.metric("Movies", f"{movies['movieId'].nunique():,}")
    metric_3.metric("Ratings", f"{len(ratings):,}")
    metric_4.metric("Selected User Ratings", f"{len(user_history):,}")

    st.subheader(f"User {selected_user} profile")
    liked_movies = user_history[user_history["rating"] >= min_rating]
    history_col, liked_col = st.columns(2)

    with history_col:
        st.markdown("**Recently highest-rated movies**")
        st.dataframe(user_history.head(15), use_container_width=True, hide_index=True)

    with liked_col:
        st.markdown(f"**Movies rated at least {min_rating}**")
        st.dataframe(liked_movies.head(15), use_container_width=True, hide_index=True)

    st.subheader("At-a-glance method quality")
    preview_rows: list[dict[str, float | str]] = []
    for method_name, frame in method_frames.items():
        score_column = next(
            (
                candidate
                for candidate in ("predicted_rating", "content_score", "avg_rating")
                if candidate in frame.columns
            ),
            None,
        )
        mean_score = float(frame[score_column].head(top_n).mean()) if score_column else 0.0
        preview_rows.append(
            {
                "Method": method_name,
                "Top-N Count": int(len(frame.head(top_n))),
                "Mean Score": round(mean_score, 3),
            }
        )
    preview_df = pd.DataFrame(preview_rows).sort_values(by="Mean Score", ascending=False)
    st.bar_chart(preview_df.set_index("Method")["Mean Score"])
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    st.subheader("Recommendations by method")
    rec_tab_1, rec_tab_2, rec_tab_3, rec_tab_4, rec_tab_5 = st.tabs(
        [
            "Popularity",
            "User-User CF",
            "Item-Item CF",
            "Content-Based",
            "Matrix Factorization",
        ]
    )

    with rec_tab_1:
        st.caption(METHOD_DESCRIPTIONS["Popularity"])
        if ui_mode == "Cards":
            _display_recommendation_cards(method_frames["Popularity"])
        else:
            st.dataframe(method_frames["Popularity"], use_container_width=True, hide_index=True)

    with rec_tab_2:
        st.caption(METHOD_DESCRIPTIONS["User-User CF"])
        if ui_mode == "Cards":
            _display_recommendation_cards(method_frames["User-User CF"])
        else:
            st.dataframe(method_frames["User-User CF"], use_container_width=True, hide_index=True)

    with rec_tab_3:
        st.caption(METHOD_DESCRIPTIONS["Item-Item CF"])
        if ui_mode == "Cards":
            _display_recommendation_cards(method_frames["Item-Item CF"])
        else:
            st.dataframe(method_frames["Item-Item CF"], use_container_width=True, hide_index=True)

    with rec_tab_4:
        st.caption(METHOD_DESCRIPTIONS["Content-Based"])
        if ui_mode == "Cards":
            _display_recommendation_cards(method_frames["Content-Based"])
        else:
            st.dataframe(method_frames["Content-Based"], use_container_width=True, hide_index=True)

    with rec_tab_5:
        st.caption(METHOD_DESCRIPTIONS["Matrix Factorization"])
        if ui_mode == "Cards":
            _display_recommendation_cards(method_frames["Matrix Factorization"])
        else:
            st.dataframe(method_frames["Matrix Factorization"], use_container_width=True, hide_index=True)

    if show_evaluation:
        st.subheader("Model comparison")
        with st.spinner("Computing evaluation metrics..."):
            evaluation_results = prepare_evaluation(ratings, movies, tags)
        st.dataframe(evaluation_results, use_container_width=True, hide_index=True)
        precision_column = next((col for col in evaluation_results.columns if col.startswith("precision@")), None)
        if precision_column:
            chart_df = (
                evaluation_results[["model", precision_column]]
                .rename(columns={"model": "Model", precision_column: "Precision"})
                .set_index("Model")
            )
            st.bar_chart(chart_df)
        st.caption("Evaluation uses a holdout split and a sample of 150 users.")


if __name__ == "__main__":
    main()
