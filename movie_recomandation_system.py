import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Inject peach-themed CSS for the app
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
        """, unsafe_allow_html=True
    )

@st.cache_data
def load_data():
    # Load Excel dataset with raw string path; adjust path if needed
    df = pd.read_excel(r"C:\Users\admin\Downloads\modified_full_movies_dataset.xls")
    return df

def main():
    # Page config with browser tab title
    st.set_page_config(page_title="Welcome to Movie Recommendation System", layout="centered")
    local_css()

    st.title("🎬 Movie Recommendation System")

    df = load_data()

    # All genre columns except 'title'
    data = df.drop(columns=['title'])
    genre_cols = list(data.columns)

    st.markdown("### Select your favorite genres:")

    # Two column layout for checkboxes
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

            user_vector = np.array([1 if g in selected_genres else 0 for g in genre_cols]).reshape(1, -1)
            distances, indices = knn.kneighbors(user_vector)

            recommended = df.iloc[indices[0]]['title'].values

            st.markdown("### Top 5 Recommended Movies:")
            for i, movie in enumerate(recommended, 1):
                st.write(f"{i}. {movie}")

if __name__ == "__main__":
    main()
