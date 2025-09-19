import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import imdb
import logging

# Setup logging to console for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_movie_details_with_logging(title):
    ia = imdb.IMDb()
    try:
        logging.info(f"Searching IMDb for movie: {title}")
        results = ia.search_movie(title)
        if not results:
            logging.warning("No results found for the movie")
            return "No description found.", None, "N/A"
        
        logging.info(f"Found {len(results)} results. Selecting the first result.")
        movie = results[0]
        logging.debug(f"Selected Movie ID: {movie.movieID}, Title: {movie}")

        movie_details = ia.get_movie(movie.movieID)
        ia.update(movie_details, info=['main', 'plot', 'vote details'])
        logging.debug(f"Fetched movie details keys: {movie_details.keys()}")

        desc = movie_details.get('plot outline') or "No description found."
        img_url = movie_details.get('cover url', None)
        rating = movie_details.get('rating', 'N/A')

        logging.info(f"Description length: {len(desc)}, Image URL: {img_url}, Rating: {rating}")
        return desc, img_url, rating

    except Exception as e:
        logging.error(f"Exception while fetching movie details: {e}")
        return "No description found.", None, "N/A"

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
            
            descriptions, images, ratings = [], [], []
            for title in rec_df['title']:
                desc, img_url, rating = get_movie_details_with_logging(title)
                descriptions.append(desc)
                images.append(img_url)
                try:
                    ratings.append(float(rating) if rating != 'N/A' else 0)
                except:
                    ratings.append(0)
            
            rec_df['imdb_rating'] = ratings
            rec_df['desc'] = descriptions
            rec_df['img_url'] = images
            
            sorted_rec = rec_df.sort_values('imdb_rating', ascending=False).head(5)
            
            st.markdown("### Top 5 Recommended Movies:")
            for _, row in sorted_rec.iterrows():
                st.subheader(row['title'])
                if row['img_url']:
                    st.image(row['img_url'], width=200)
                st.write(f"**Description:** {row['desc']}")
                st.write(f"**IMDb Rating:** ⭐ {row['imdb_rating']}/10")
                st.write("---")

if __name__ == "__main__":
    main()
