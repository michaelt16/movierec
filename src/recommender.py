import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.is_trained = False
        
        # Flag to detect if NLTK is available
        self.nltk_available = True
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            
            # Check if NLTK data is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading required NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
        except ImportError:
            logger.warning("NLTK not available, falling back to basic text processing")
            self.nltk_available = False
    
    def preprocess_text(self, text):
        """Preprocess text data for better NLP analysis."""
        if not isinstance(text, str):
            return ""
            
        if self.nltk_available:
            try:
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                
                # Tokenize
                tokens = word_tokenize(text.lower())
                
                # Remove stopwords and lemmatize
                stop_words = set(stopwords.words('english'))
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token.isalnum() and token not in stop_words]
                
                return " ".join(tokens)
            except Exception as e:
                logger.warning(f"Error in NLTK processing: {e}, falling back to basic")
                # Fall back to basic processing
                return " ".join([word.lower() for word in text.split() if len(word) > 2])
        else:
            # Basic text processing without NLTK
            return " ".join([word.lower() for word in text.split() if len(word) > 2])
    
    def create_movie_features(self, movies_data):
        """Create feature matrix from movie data."""
        start_time = time.time()
        logger.info("Starting feature creation...")
        
        # Convert movies data to DataFrame if it's not already
        self.movies_df = pd.DataFrame(movies_data) if not isinstance(movies_data, pd.DataFrame) else movies_data
        
        logger.info(f"Creating features for {len(self.movies_df)} movies")
        
        # Check if important columns exist
        required_columns = ['id', 'title']
        for col in required_columns:
            if col not in self.movies_df.columns:
                logger.error(f"Required column '{col}' missing from movie data")
                raise ValueError(f"Required column '{col}' missing from movie data")
        
        # Print columns to debug
        logger.info(f"Available columns: {self.movies_df.columns.tolist()}")
        
        # Combine relevant features - safely get values that might not exist
        self.movies_df['combined_features'] = self.movies_df.apply(
            lambda x: (
                f"{x.get('title', '')} "
                f"{x.get('overview', '')} "
                f"{x.get('genres', '')} "
                f"{' '.join(str(x.get('genre_ids', [])))}"
            ), 
            axis=1
        )
        
        # Debug the first few combined features
        logger.info(f"Sample combined features: {self.movies_df['combined_features'].head(1).values}")
        
        # Preprocess the combined features
        logger.info("Preprocessing text features...")
        self.movies_df['processed_features'] = self.movies_df['combined_features'].apply(self.preprocess_text)
        
        # Create TF-IDF matrix
        logger.info("Creating TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['processed_features'])
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Set trained flag
        self.is_trained = True
        
        # Record training time
        end_time = time.time()
        logger.info(f"Feature creation completed in {end_time - start_time:.2f} seconds")
        
        return self
    
    def get_recommendations(self, movie_id, n_recommendations=5):
        """Get movie recommendations based on similarity."""
        if not self.is_trained or self.tfidf_matrix is None:
            logger.error("Model not trained. Call create_movie_features first.")
            raise ValueError("Model not trained. Call create_movie_features first.")
        
        logger.info(f"Getting recommendations for movie ID: {movie_id}")
            
        try:
            # Check if movie_id is in the dataframe
            if movie_id not in self.movies_df['id'].values:
                logger.warning(f"Movie ID {movie_id} not found in the dataset")
                return []
            
            # Find the movie index
            movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]
            logger.info(f"Found movie at index: {movie_idx}")
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(
                self.tfidf_matrix[movie_idx:movie_idx+1], 
                self.tfidf_matrix
            ).flatten()
            
            # Get indices of movies with highest similarity scores
            similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
            logger.info(f"Found {len(similar_indices)} similar movies")
            
            # Return recommended movies
            recommendations = self.movies_df.iloc[similar_indices]
            return recommendations.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def get_genre_based_recommendations(self, genres, n_recommendations=5):
        """Get recommendations based on genre preferences."""
        if not self.is_trained or self.movies_df is None:
            logger.error("No movie data available. Call create_movie_features first.")
            raise ValueError("No movie data available. Call create_movie_features first.")
        
        logger.info(f"Getting genre-based recommendations for genres: {genres}")
            
        try:
            # Ensure genres column exists
            if 'genres' not in self.movies_df.columns:
                logger.warning("'genres' column not found, trying to use genre_ids")
                return []
            
            # Calculate genre match scores
            genre_scores = self.movies_df['genres'].apply(
                lambda x: sum(genre.lower() in str(x).lower() for genre in genres) if x is not None else 0
            )
            
            # Get indices of movies with highest genre matches
            top_indices = genre_scores.sort_values(ascending=False).head(n_recommendations).index
            
            # Return recommended movies
            recommendations = self.movies_df.iloc[top_indices]
            logger.info(f"Found {len(recommendations)} genre-based recommendations")
            return recommendations.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {str(e)}")
            return []