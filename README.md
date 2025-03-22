# MovieMind: AI-Powered Movie Recommender

MovieMind is an intelligent movie recommendation system that combines the power of Natural Language Processing (NLP) and machine learning to provide personalized movie suggestions based on user preferences and movie content analysis.

## Features

- Content-based movie recommendations using NLP
- Integration with TMDB API for up-to-date movie data
- User-friendly interface built with Streamlit
- Sentiment analysis of movie descriptions
- Similar movie suggestions based on plot, genre, and cast

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your TMDB API key:
```
TMDB_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## Project Structure

```
movie-recommender/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── recommender.py         # Recommendation engine
│   └── tmdb_client.py        # TMDB API client
├── static/
│   └── styles/               # CSS and other static files
├── templates/
│   └── components/           # Reusable UI components
├── .env                      # Environment variables
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Technologies Used

- Python 3.8+
- Streamlit
- TMDB API
- scikit-learn
- NLTK
- Transformers (Hugging Face)
- PyTorch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 