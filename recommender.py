
"""
Content-Based Movie Recommendation System
Assignment 20
"""

import ast
import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------
# 1. DATASET PATH
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "tmdb_5000_movies.csv"


# ----------------------------------------
# 2. EXTRACT GENRE NAMES
# ----------------------------------------

def extract_genres(value):

    if pd.isna(value):
        return ""

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):

            return " ".join(
                item.get("name", "")
                for item in parsed
                if isinstance(item, dict)
            )

    except (ValueError, SyntaxError):
        pass

    return str(value)


# ----------------------------------------
# 3. TEXT PREPROCESSING
# ----------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove stopwords
    words = [
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


# ----------------------------------------
# 4. LOAD AND PROCESS DATASET
# ----------------------------------------

def load_recommender(csv_path=DATA_PATH):

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Download tmdb_5000_movies.csv from Kaggle "
            "and put it inside the data folder."
        )

    # Load CSV
    movies = pd.read_csv(csv_path)

    # Check required columns
    required = {"title", "overview"}

    missing = required - set(movies.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ------------------------------------
    # TASK 1: DATASET EXPLORATION
    # ------------------------------------

    print("\n========== DATASET EXPLORATION ==========")

    print("Dataset shape:")
    print(movies.shape)

    print("\nColumn names:")
    print(movies.columns.tolist())

    print("\nFirst five rows:")
    print(movies.head())

    print("\nDataset information:")
    movies.info()

    print("\nMissing values:")
    print(movies.isnull().sum())

    # ------------------------------------
    # TASK 2: MISSING VALUE HANDLING
    # ------------------------------------

    movies = movies.copy()

    movies["title"] = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    movies["overview"] = (
        movies["overview"]
        .fillna("")
        .astype(str)
    )

    # Extract genre names
    if "genres" in movies.columns:

        movies["genres_text"] = (
            movies["genres"].apply(extract_genres)
        )

    else:

        movies["genres_text"] = ""

    # Remove records without movie titles
    movies = movies[
        movies["title"] != ""
    ].reset_index(drop=True)

    # ------------------------------------
    # TASK 3: COMBINE TEXT
    # ------------------------------------

    movies["recommendation_text"] = (
        movies["overview"]
        + " "
        + movies["genres_text"]
    )

    # Apply preprocessing
    movies["clean_text"] = (
        movies["recommendation_text"]
        .apply(clean_text)
    )

    print("\nCleaned text:")
    print(
        movies[
            ["title", "overview", "clean_text"]
        ].head()
    )

    # ------------------------------------
    # TASK 4: TF-IDF VECTORIZATION
    # ------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["clean_text"]
    )

    print("\n========== TF-IDF ==========")

    print(
        "TF-IDF matrix shape:",
        tfidf_matrix.shape
    )

    # ------------------------------------
    # TASK 5: COSINE SIMILARITY
    # ------------------------------------

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    ).astype("float32")

    print("\n========== SIMILARITY ==========")

    print(
        "Similarity matrix shape:",
        similarity_matrix.shape
    )

    return (
        movies,
        tfidf_matrix,
        similarity_matrix
    )


# ----------------------------------------
# 6. RECOMMENDATION FUNCTION
# ----------------------------------------

def recommend(
    item_name,
    movies,
    similarity_matrix,
    top_n=5
):

    empty = pd.DataFrame(
        columns=[
            "title",
            "overview",
            "similarity_score"
        ]
    )

    if not item_name or not str(item_name).strip():
        return empty

    # Find exact movie title
    matches = movies.index[
        movies["title"].str.casefold()
        == str(item_name).casefold()
    ].tolist()

    # Partial title matching if exact match fails
    if not matches:

        matches = movies.index[
            movies["title"].str.contains(
                re.escape(str(item_name)),
                case=False,
                na=False
            )
        ].tolist()

    if not matches:
        return empty

    # Get selected movie index
    selected_index = matches[0]

    # Get similarity scores
    scores = list(
        enumerate(
            similarity_matrix[selected_index]
        )
    )

    # Exclude selected movie itself
    scores = [
        (idx, score)
        for idx, score in scores
        if idx != selected_index
    ]

    # Sort by similarity score
    scores.sort(
        key=lambda pair: pair[1],
        reverse=True
    )

    # Select top N movies
    top_matches = scores[:max(1, int(top_n))]

    indices = [
        idx for idx, score in top_matches
    ]

    values = [
        float(score) for idx, score in top_matches
    ]

    results = movies.loc[
        indices,
        ["title", "overview"]
    ].copy()

    results["similarity_score"] = values

    return results.reset_index(drop=True)


# ----------------------------------------
# 7. TEST RECOMMENDATIONS
# ----------------------------------------

if __name__ == "__main__":

    movies, tfidf, similarity = load_recommender()

    test_movies = [
        "The Dark Knight",
        "Avatar",
        "Toy Story"
    ]

    for title in test_movies:

        print(f"\nRecommendations for: {title}")

        results = recommend(
            title,
            movies,
            similarity,
            top_n=5
        )

        if results.empty:

            print("Movie not found. Choose a title from the CSV.")

        else:

            print(
                results[
                    ["title", "similarity_score"]
                ].to_string(index=False)
            )
