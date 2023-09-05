import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript, clear_output

# Load the dataset with explicit character encoding (Make sure to upload 'movies.csv' to your Colab environment)
data = pd.read_csv('/content/movies (1).csv', encoding='latin1')

# Clean movie titles by removing leading/trailing spaces and making them lowercase
data['title'] = data['title'].str.strip().str.lower()

# Perform TF-IDF vectorization on the 'genres' column
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['genres'])

# Calculate cosine similarity between movies based on their genres
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on user input using fuzzy string matching
def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Clean the user input by removing leading/trailing spaces and making it lowercase
    movie_title = movie_title.strip().lower()
    
    # Use fuzzy string matching to find the closest matching titles in the dataset
    similarity_scores = [(title, fuzz.partial_ratio(movie_title, title.lower())) for title in data['title']]
    
    # Filter titles with similarity scores above a threshold (adjust as needed)
    threshold = 70  # Adjust the threshold as needed
    matched_titles = [title for title, score in similarity_scores if score >= threshold]
    
    if not matched_titles:
        return ["No similar movie found in the database. Please try another one."]
    
    # Get the indices of the matched movie titles
    indices = [idx for idx, title in enumerate(data['title']) if title in matched_titles]

    # Calculate the average cosine similarity scores for matched movies
    avg_cosine_scores = cosine_sim[indices].mean(axis=0)

    # Sort the movies based on the average similarity scores
    movie_indices = avg_cosine_scores.argsort()[::-1][1:11]

    # Return the top 10 recommended movie titles
    return data['title'].iloc[movie_indices].tolist()

# Function to handle the recommendation process and open a new tab
def recommend_movies_ui(button):
    user_input = input_movie.value  # Get user input
    recommendations = recommend_movies(user_input)  # Get movie recommendations

    # Clear previous output
    clear_output()

    # Display recommendations in a new tab
    display(HTML("<h2>Recommended Movies</h2>"))
    for movie in recommendations:
        display(HTML(f"<p>{movie}</p>"))

# Create UI elements
input_movie = widgets.Text(placeholder="Enter a movie title")
search_button = widgets.Button(description="Search")
search_button.on_click(recommend_movies_ui)

# Display UI
display(input_movie, search_button)
