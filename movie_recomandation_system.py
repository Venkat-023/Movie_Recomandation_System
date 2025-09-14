import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load your movie dataset (adjust the path)
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\admin\\Downloads\\modified_full_movies_dataset.xls")  # your full dataset
    return df

def main():
    st.title("Movie Recommendation System")

    df = load_data()
    data=df.drop(columns=['title'])
    genre_cols =data.columns # use your full genre list

    selected_genres = []
    st.write("Select genres:")
    for genre in genre_cols:
        if st.checkbox(genre):
            selected_genres.append(genre)

    if st.button("Recommend"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            knn = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn.fit(df[genre_cols])

            # Create user vector
            user_vector = np.array([1 if genre in selected_genres else 0 for genre in genre_cols]).reshape(1, -1)

            # Get recommendations
            distances, indices = knn.kneighbors(user_vector)
            recommended = df.iloc[indices[0]]['title']

            st.write("Top 5 movies recommended:")
            for movie in recommended:
                st.write(movie)

if __name__ == "__main__":
    main()
