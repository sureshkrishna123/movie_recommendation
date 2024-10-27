import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the saved similarity matrix model
model_path = "movie_recommender_model.pkl"
with open(model_path, 'rb') as f:
    similarity_matrix = pickle.load(f)

# 2. Load the movies dataset
movies_path = "tmdb_5000_movies.csv"
movies_df = pd.read_csv(movies_path)
movies_df = movies_df[['id', 'title', 'genres']].dropna()  # Ensure no missing titles

# 3. Extract genres into a list for each movie
def extract_genres(genre_str):
    """Convert the 'genres' column (JSON-like) to a list of genre names."""
    try:
        genres = eval(genre_str)
        return [genre['name'] for genre in genres]
    except:
        return []

movies_df['genres_list'] = movies_df['genres'].apply(extract_genres)

# 4. Get all unique genres
all_genres = set([genre for genres in movies_df['genres_list'] for genre in genres])

# 5. Function to get movie recommendations by name
def get_recommendations_by_name(movie_name, similarity_matrix, top_n=10):
    """Recommend top N movies based on the given movie name."""
    movie_idx = movies_df[movies_df['title'].str.lower() == movie_name.lower()].index
    if len(movie_idx) == 0:
        return None  # Movie not found

    movie_idx = movie_idx[0]
    similarity_scores = similarity_matrix[movie_idx]
    top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]  # Exclude the input movie

    return movies_df.iloc[top_indices][['title']]

# 6. Streamlit App UI
st.title("Movie Recommender System")

# Sidebar with two options: Recommend by Movie or List by Genre
option = st.sidebar.radio(
    "Choose an option:",
    ("Recommend Movies by Name", "Display Movies by Genre")
)

if option == "Recommend Movies by Name":
    # Input: Movie Name
    movie_name = st.text_input("Enter a movie name:")

    # Input: Number of Recommendations
    top_n = st.slider("Number of Recommendations", 1, 20, 5)

    if st.button("Get Recommendations"):
        if movie_name:
            recommendations = get_recommendations_by_name(movie_name, similarity_matrix, top_n)

            if recommendations is not None:
                st.write(f"Top {top_n} movies recommended for **{movie_name}**:")
                st.table(recommendations)
            else:
                st.error(f"Movie '{movie_name}' not found in the dataset. Please try another movie name.")
        else:
            st.warning("Please enter a movie name.")

elif option == "Display Movies by Genre":
    # Dropdown to select a genre
    selected_genre = st.selectbox("Select a genre:", sorted(all_genres))

    # Input: Number of Movies to Display
    num_movies = st.slider("Number of Movies to Display", 1, 20, 5)

    if st.button("Show Movies"):
        # Filter movies by selected genre
        filtered_movies = movies_df[movies_df['genres_list'].apply(lambda x: selected_genre in x)]

        if not filtered_movies.empty:
            st.write(f"Top {num_movies} movies in the genre **{selected_genre}**:")
            st.table(filtered_movies[['title']].head(num_movies))
        else:
            st.write(f"No movies found in the genre **{selected_genre}**.")
