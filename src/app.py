import streamlit as st
import pandas as pd
from tmdb_client import TMDBClient
from recommender import MovieRecommender
import os
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="MovieMind - AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1DB954;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #2C2C2C;
        margin: 1rem 0;
    }
    .movie-title {
        color: #1DB954;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .poster-img {
        max-width: 85%;
        margin: 0 auto;
        display: block;
        transition: transform 0.3s ease;
        border-radius: 5px;
    }
    .poster-img:hover {
        transform: scale(1.1);
        z-index: 10;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)
# App title
st.title("MovieMind: AI-Powered Movie Recommendations")
st.markdown("---")

# Initialize session state variables
if 'movie_data' not in st.session_state:
    st.session_state.movie_data = {}

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'recommender_trained' not in st.session_state:
    st.session_state.recommender_trained = False
    
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
    
if 'training_started_at' not in st.session_state:
    st.session_state.training_started_at = None
    
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None

# Initialize TMDB client
tmdb_client = TMDBClient()

# Sidebar
st.sidebar.title("Preferences")
recommendation_type = st.sidebar.selectbox(
    "How would you like to get recommendations?",
    ["Search by Movie", "Browse Popular", "By Genre"]
)

# Debug toggle in sidebar
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Function to train the recommender model
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def train_recommender_model(movies_data):
    """Create and train the recommender model with the provided data."""
    try:
        recommender = MovieRecommender()
        
        # Convert to DataFrame if needed
        if not isinstance(movies_data, pd.DataFrame):
            movies_df = pd.DataFrame(movies_data)
        else:
            movies_df = movies_data
            
        # Train the recommender with this data
        recommender.create_movie_features(movies_df)
        return recommender, movies_df
    except Exception as e:
        logger.error(f"Error in training recommender: {str(e)}")
        return None, None

# Function to fetch movies data
def fetch_movies_data(debug=False):
    """Fetch movie data from the TMDB API."""
    try:
        # Get a collection of movies to train the recommender with
        popular_movies = tmdb_client.get_popular_movies(page=1)
        movies_data = popular_movies.get('results', [])
        
        if debug:
            st.sidebar.write(f"Fetched {len(movies_data)} movies from first page")
        
        # Fetch additional pages for more training data
        for page in range(2, 4):  # Getting 3 pages of popular movies
            more_movies = tmdb_client.get_popular_movies(page=page)
            more_data = more_movies.get('results', [])
            movies_data.extend(more_data)
            
            if debug:
                st.sidebar.write(f"Fetched {len(more_data)} movies from page {page}")
        
        # Convert to DataFrame
        movies_df = pd.DataFrame(movies_data)
        
        if debug:
            st.sidebar.write(f"Total movies for training: {len(movies_df)}")
            st.sidebar.write(f"DataFrame columns: {movies_df.columns.tolist()}")
        
        # Add genres as strings
        if 'genre_ids' in movies_df.columns:
            # Get genre mapping
            genres = tmdb_client.get_movie_genres()
            genre_dict = {genre['id']: genre['name'] for genre in genres.get('genres', [])}
            
            # Add a genres string column
            movies_df['genres'] = movies_df['genre_ids'].apply(
                lambda ids: ' '.join([genre_dict.get(id, '') for id in ids]) if isinstance(ids, list) else ''
            )
            
            if debug:
                st.sidebar.write("Added genre names to movies")
                
        return movies_df
    except Exception as e:
        logger.error(f"Error fetching movie data: {str(e)}")
        return None

# Check if we need to train the recommender
if not st.session_state.recommender_trained and not st.session_state.training_in_progress:
    st.session_state.training_in_progress = True
    st.session_state.training_started_at = time.time()
    
    # Show a progress message
    training_progress = st.progress(0)
    training_status = st.empty()
    training_status.info("Initializing recommender system... Please wait. This may take a minute.")
    
    # Fetch movie data
    movies_df = fetch_movies_data(debug_mode)
    
    if movies_df is not None:
        # Update progress
        training_progress.progress(30)
        training_status.info("Movie data fetched. Training recommendation model...")
        
        # Train the recommender
        recommender, trained_df = train_recommender_model(movies_df)
        
        if recommender is not None:
            # Store in session state
            st.session_state.recommender = recommender
            st.session_state.movies_df = trained_df
            st.session_state.recommender_trained = True
            
            # Update progress
            training_progress.progress(100)
            training_status.success("Recommender system initialized successfully!")
            
            if debug_mode:
                st.sidebar.success("Recommender system initialized successfully")
        else:
            training_progress.progress(100)
            training_status.error("Failed to train recommender model. Please refresh the page to try again.")
    else:
        training_progress.progress(100)
        training_status.error("Failed to fetch movie data. Please check your API connection and try again.")
    
    # Clear the training flag
    st.session_state.training_in_progress = False

# Check if training is still in progress or timed out
if st.session_state.training_in_progress:
    # If training has been going on for more than 3 minutes, consider it timed out
    current_time = time.time()
    training_duration = current_time - st.session_state.training_started_at
    
    if training_duration > 180:  # 3 minutes in seconds
        st.warning("Training is taking longer than expected. You can continue browsing while it completes.")
        st.session_state.training_in_progress = False
    else:
        st.info("Initializing recommender system... Please wait.")

# Main content - only show when not in initial training
if not st.session_state.training_in_progress or (st.session_state.training_in_progress and time.time() - st.session_state.training_started_at > 180):
    if recommendation_type == "Search by Movie":
        search_query = st.text_input("Search for a movie:", "")
        
        if search_query:
            with st.spinner('Searching for movies...'):
                search_results = tmdb_client.search_movies(search_query)
                
                if search_results.get('results', []):
                    st.subheader("Search Results")
                    cols = st.columns(3)
                    
                    for idx, movie in enumerate(search_results['results'][:6]):
                        col = cols[idx % 3]
                        with col:
                            st.markdown(f"### {movie['title']}")
                            if movie.get('poster_path'):
                                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                st.image(poster_url, use_container_width=True)
                            else:
                                st.write("No poster available")
                                
                            # Display rating and release date
                            st.markdown(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
                            st.markdown(f"**Release Date:** {movie.get('release_date', 'N/A')}")
                            
                            # Display movie overview in an expander
                            if movie.get('overview'):
                                with st.expander("Overview"):
                                    st.write(movie['overview'])
                            
                            # Button to get recommendations
                            recommendation_button = st.button("Get Recommendations", key=f"rec_{movie['id']}")
                            
                            if recommendation_button:
                                if not st.session_state.recommender_trained:
                                    st.warning("Recommender system is still initializing. Please wait a moment and try again.")
                                else:
                                    # Store the full movie object
                                    st.session_state.selected_movie = movie['id']
                                    
                                    # We need detailed movie data
                                    movie_details = tmdb_client.get_movie_details(movie['id'])
                                    st.session_state.movie_data = movie_details
                                    
                                    if debug_mode:
                                        st.sidebar.write(f"Selected movie: {movie['title']} (ID: {movie['id']})")
                                        if movie_details:
                                            st.sidebar.write("Successfully fetched movie details")
                                        else:
                                            st.sidebar.error("Failed to fetch movie details")
                                    
                                    # Use st.rerun() to refresh the page
                                    st.rerun()

    elif recommendation_type == "Browse Popular":
        with st.spinner('Loading popular movies...'):
            popular_movies = tmdb_client.get_popular_movies()
            
            st.subheader("Popular Movies")
            cols = st.columns(3)
            
            for idx, movie in enumerate(popular_movies.get('results', [])[:6]):
                col = cols[idx % 3]
                with col:
                    st.markdown(f"### {movie['title']}")
                    if movie.get('poster_path'):
                        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                        st.image(poster_url, use_container_width=True)
                    
                    # Display movie details
                    st.markdown(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
                    st.markdown(f"**Release Date:** {movie.get('release_date', 'N/A')}")
                    
                    if movie.get('overview'):
                        with st.expander("Overview"):
                            st.write(movie['overview'])
                            
                    # Button to get recommendations
                    recommendation_button = st.button("Get Recommendations", key=f"pop_{movie['id']}")
                    
                    if recommendation_button:
                        if not st.session_state.recommender_trained:
                            st.warning("Recommender system is still initializing. Please wait a moment and try again.")
                        else:
                            st.session_state.selected_movie = movie['id']
                            movie_details = tmdb_client.get_movie_details(movie['id'])
                            st.session_state.movie_data = movie_details
                            st.rerun()

    else:  # By Genre
        with st.spinner('Loading genres...'):
            genres = tmdb_client.get_movie_genres()
            selected_genres = st.sidebar.multiselect(
                "Select your favorite genres:",
                [genre['name'] for genre in genres.get('genres', [])]
            )
            
            if selected_genres:
                if not st.session_state.recommender_trained:
                    st.warning("Recommender system is still initializing. Please wait a moment and try again.")
                else:
                    st.subheader("Recommended Movies Based on Genres")
                    try:
                        # Use the cached recommender
                        recommender = st.session_state.recommender
                        recommendations = recommender.get_genre_based_recommendations(selected_genres)
                        
                        if recommendations:
                            cols = st.columns(3)
                            for idx, movie in enumerate(recommendations[:6]):
                                col = cols[idx % 3]
                                with col:
                                    st.markdown(f"### {movie['title']}")
                                    if movie.get('poster_path'):
                                        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                        st.image(poster_url, use_container_width=True)
                                    st.markdown(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
                                    
                                    # Display movie overview in an expander
                                    if movie.get('overview'):
                                        with st.expander("Overview"):
                                            st.write(movie['overview'])
                        else:
                            st.info("No recommendations found for the selected genres. Try selecting different genres.")
                            
                            if debug_mode:
                                st.sidebar.warning("No recommendations found for the selected genres")
                    except Exception as e:
                        error_msg = f"Error getting genre recommendations: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
                        if debug_mode:
                            st.sidebar.error(error_msg)

# Show recommendations if a movie is selected
if 'selected_movie' in st.session_state and st.session_state.recommender_trained:
    with st.spinner('Getting recommendations...'):
        movie_id = st.session_state.selected_movie
        movie_data = st.session_state.movie_data
        
        # Debug information
        if debug_mode:
            st.sidebar.write(f"Getting recommendations for movie ID: {movie_id}")
            st.sidebar.write(f"Movie data available: {bool(movie_data)}")
        
        try:
            # Try getting recommendations
            recommender = st.session_state.recommender
            recommendations = recommender.get_recommendations(movie_id)
            
            if debug_mode:
                st.sidebar.write(f"Received {len(recommendations)} recommendations")
            
            if not recommendations:
                # The movie might not be in our training set
                # Add it to the recommender and try again
                if movie_data:
                    if debug_mode:
                        st.sidebar.write("Adding selected movie to recommender and retraining")
                    
                    # Create a small DataFrame with just this movie
                    new_movie = pd.DataFrame([movie_data])
                    
                    # Get current movie data
                    current_df = st.session_state.movies_df
                    
                    # Add the new movie
                    if 'genre_ids' in new_movie.columns and 'genres' not in new_movie.columns:
                        # Convert genre IDs to genre names
                        genres = tmdb_client.get_movie_genres()
                        genre_dict = {genre['id']: genre['name'] for genre in genres.get('genres', [])}
                        
                        # Add genres string column
                        new_movie['genres'] = new_movie['genre_ids'].apply(
                            lambda ids: ' '.join([genre_dict.get(id, '') for id in ids]) if isinstance(ids, list) else ''
                        )
                    
                    # Combine with existing data
                    updated_df = pd.concat([current_df, new_movie], ignore_index=True)
                    
                    # Retrain recommender
                    st.session_state.movies_df = updated_df
                    recommender.create_movie_features(updated_df)
                    
                    # Try getting recommendations again
                    recommendations = recommender.get_recommendations(movie_id)
                    
                    if debug_mode:
                        st.sidebar.write(f"After retraining, received {len(recommendations)} recommendations")
            
            if recommendations:
                st.subheader(f"Because you liked {movie_data.get('title', 'this movie')}")
                
                cols = st.columns(3)
                for idx, rec in enumerate(recommendations):
                    col = cols[idx % 3]
                    with col:
                        st.markdown(f"### {rec['title']}")
                        if rec.get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w500{rec['poster_path']}"
                            st.image(poster_url, use_container_width=True)
                        st.markdown(f"**Rating:** {rec.get('vote_average', 'N/A')}/10")
                        
                        # Display movie overview in an expander
                        if rec.get('overview'):
                            with st.expander("Overview"):
                                st.write(rec['overview'])
            else:
                st.info("No similar movies found. Try searching for a different movie.")
                
                if debug_mode:
                    st.sidebar.error("Failed to get recommendations even after retraining")
                    
        except Exception as e:
            error_msg = f"Error processing recommendations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            if debug_mode:
                st.sidebar.error(error_msg)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ by MovieMind | Powered by TMDB API and AI",
    unsafe_allow_html=True
)