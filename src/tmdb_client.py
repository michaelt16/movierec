import os
import requests
from dotenv import load_dotenv

class TMDBClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        
    def _make_request(self, endpoint, params=None):
        """Make a request to the TMDB API."""
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_popular_movies(self, page=1):
        """Get popular movies."""
        return self._make_request('/movie/popular', {'page': page})
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a specific movie."""
        return self._make_request(f'/movie/{movie_id}', {
            'append_to_response': 'credits,keywords,recommendations'
        })
    
    def search_movies(self, query, page=1):
        """Search for movies by title."""
        return self._make_request('/search/movie', {
            'query': query,
            'page': page
        })
    
    def get_movie_credits(self, movie_id):
        """Get cast and crew information for a movie."""
        return self._make_request(f'/movie/{movie_id}/credits')
    
    def get_movie_keywords(self, movie_id):
        """Get keywords associated with a movie."""
        return self._make_request(f'/movie/{movie_id}/keywords')
    
    def get_similar_movies(self, movie_id, page=1):
        """Get similar movies based on genres and keywords."""
        return self._make_request(f'/movie/{movie_id}/similar', {'page': page})
    
    def get_movie_genres(self):
        """Get list of official genres."""
        return self._make_request('/genre/movie/list') 