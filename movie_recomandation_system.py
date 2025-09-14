import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def local_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: #FFE5B4;
            color: #5A3E1B;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1, h2, h3 {
            color: #D2691E;
            font-weight: bold;
        }
        div.stButton > button {
            background-color: #FFB380;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #FF8533;
            cursor: pointer;
        }
        label[data-baseweb="checkbox"] > div {
            color: #5A3E1B;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def load_data():
    df = pd.read_csv("final_movies.csv")
    return df

def main():
    st.set_page_config(
        page_title="Welcome to Movie Recommendation System",
        layout="centered",
    )
    local_css()

    st.title("🎬 Movie Recommendation System")

    df = load_data()

    # Debug: show columns loaded and sample data
    st.write("Columns loaded:", df.columns.tolist())
    st.write(df.head())

    if 'title' not in df.columns:
        st.error("Error: 'title' column not found in the dataset.")
        st.stop()

    # Drop the title column for genre features
    genre_cols = [
    "adventure",
    "animation",
    "comedy",
    "fantasy",
    "romance",
    "children",
    "drama",
    "documentary",
    "crime",
    "sci-fi",
    "horror",
    "mystery",
    "war",
    "thriller",
    "action"
]


    st.markdown("### Select your favorite genres:")

    col1, col2 = st.columns(2)
    selected_genres = []
    half = len(genre_cols) // 2

    with col1:
        for genre in genre_cols[:half]:
            if st.checkbox(genre, key=genre):
                selected_genres.append(genre)

    with col2:
        for genre in genre_cols[half:]:
            if st.checkbox(genre, key=genre + "_2"):
                selected_genres.append(genre)

    st.write("---")

    if st.button("Recommend Movies 🎯"):
        if not selected_genres:
            st.warning("⚠️ Please select at least one genre to get recommendations.")
        else:
            knn = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn.fit(data)
            user_vector = np.array(
                [1 if genre in selected_genres else 0 for genre in genre_cols]
            ).reshape(1, -1)
            distances, indices = knn.kneighbors(user_vector)
            recommended = df.iloc[indices[0]]['title'].values

            st.markdown("### Top 5 Recommended Movies:")
            for i, movie in enumerate(recommended, 1):
                st.write(f"{i}. {movie}")

if __name__ == "__main__":
    main()

