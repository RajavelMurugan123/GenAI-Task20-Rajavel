
import streamlit as st

from recommender import (
    load_recommender,
    recommend
)


# ----------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)


# ----------------------------------------
# 2. APPLICATION TITLE
# ----------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies with similar descriptions "
    "and genres using content-based filtering."
)


# ----------------------------------------
# 3. LOAD RECOMMENDATION MODEL
# ----------------------------------------

@st.cache_resource(show_spinner="Preparing recommendation model...")
def get_model():

    return load_recommender()


try:

    movies, tfidf_matrix, similarity_matrix = get_model()

except FileNotFoundError as error:

    st.error("Movie dataset not found.")

    st.write(
        "Download tmdb_5000_movies.csv from Kaggle "
        "and place it inside the data folder."
    )

    st.code(str(error))

    st.stop()

except Exception as error:

    st.error("Unable to load recommendation model.")

    st.exception(error)

    st.stop()


# ----------------------------------------
# 4. DISPLAY DATASET INFORMATION
# ----------------------------------------

st.caption(
    f"Movies available: {len(movies):,} | "
    f"TF-IDF features: {tfidf_matrix.shape[1]:,}"
)


# ----------------------------------------
# 5. MOVIE SELECTION DROPDOWN
# ----------------------------------------

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].tolist()
)


# ----------------------------------------
# 6. SELECT NUMBER OF RECOMMENDATIONS
# ----------------------------------------

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# ----------------------------------------
# 7. RECOMMENDATION BUTTON
# ----------------------------------------

if st.button(
    "Recommend Movies",
    type="primary",
    use_container_width=True
):

    results = recommend(
        item_name=selected_movie,
        movies=movies,
        similarity_matrix=similarity_matrix,
        top_n=top_n
    )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    if results.empty:

        st.warning(
            "No recommendations found. Try another movie."
        )

    else:

        for number, movie in enumerate(
            results.itertuples(index=False),
            1
        ):

            st.markdown(
                f"### {number}. {movie.title}"
            )

            st.write(
                movie.overview or "No overview available."
            )

            st.caption(
                f"Similarity score: {movie.similarity_score:.3f}"
            )

            st.divider()


# ----------------------------------------
# 8. ABOUT SECTION
# ----------------------------------------

with st.expander("How does this application work?"):

    st.write(
        "This application uses movie overview and genre "
        "information to recommend similar movies."
    )

    st.write(
        "TF-IDF converts text into numerical vectors, "
        "and cosine similarity compares those vectors."
    )
