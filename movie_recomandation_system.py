import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import imdb

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
    df = pd.read_csv("final10.xls")
    df.columns = df.columns.str.strip().str.lower()
    return df

def get_movie_details(title):
    ia = imdb.IMDb()
    try:
        results = ia.search_movie(title)
        if results:
            movie = ia.get_movie(results[0].movieID)
            desc = movie.get('plot outline', 'No description available.')
            img_url = movie.get('cover url', None)
            rating = movie.get('rating', 'N/A')
            return desc, img_url, rating
        else:
            return 'No description found.', None, 'N/A'
    except Exception:
        return 'No description found.', None, 'N/A'

def main():
    st.set_page_config(
        page_title="Welcome to Movie Recommendation System",
        layout="centered",
    )
    local_css()
    st.title("🎬 Movie Recommendation System")
    df = load_data()
    genre_cols = [
        "adventure", "animation", "comedy", "fantasy", "romance", "children",
        "drama", "documentary", "crime", "sci-fi", "horror", "mystery", "war",
        "thriller", "action"
    ]
    data = df[genre_cols]
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
            knn = NearestNeighbors(n_neighbors=10, metric='cosine')
            knn.fit(data)
            user_vector = np.array(
                [1 if genre in selected_genres else 0 for genre in genre_cols]
            ).reshape(1, -1)
            distances, indices = knn.kneighbors(user_vector)
            rec_df = df.iloc[indices[0]].copy()
            # Get IMDb ratings for sorting
            ratings = []
            for title in rec_df['title']:
                _, _, rating = get_movie_details(title)
                try:
                    ratings.append(float(rating))
                except:
                    ratings.append(0)
            rec_df['imdb_rating'] = ratings
            sorted_rec = rec_df.sort_values('imdb_rating', ascending=False).head(5)
            st.markdown("### Top 5 Recommended Movies:")
            for i, row in sorted_rec.iterrows():
                desc, img_url, rating = get_movie_details(row['title'])
                st.subheader(f"{row['title']}")
                if img_url:
                    st.image(img_url, width=200)
                st.write(f"**Description:** {desc}")
                st.write(f"**IMDb Rating:** ⭐ {rating}/10")
                st.write("---")

if __name__ == "__main__":
    main()
