import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

app = Flask(__name__)

# Load data
df = pd.read_csv('movies.csv')

# Create features for recommendation
df['features'] = df['director'] + ' ' + df['genre'] + ' ' + df['cast'] + ' ' + df['description']

# Create similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Create movie index
indices = pd.Series(df.index, index=df['title'].str.lower())


def correct_movie_name(name):
    """Fix spelling mistakes"""
    movies = df['title'].tolist()
    # Try exact match
    if name.lower() in indices:
        return name
    # Fuzzy match
    match, score = process.extractOne(name, movies)
    return match if score > 70 else None


def get_recommendations(title, n=8):
    """Get similar movies"""
    # Correct spelling
    corrected = correct_movie_name(title)
    if not corrected:
        return None, f"Movie '{title}' not found"

    # Get recommendations
    idx = indices[corrected.lower()]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    # Prepare results
    results = []
    for i, score in scores:
        movie = df.iloc[i]
        results.append({
            'title': movie['title'],
            'director': movie['director'],
            'genre': movie['genre'],
            'year': int(movie['year']),
            'rating': movie['rating'],
            'cast': movie['cast'][:50] + '...' if len(movie['cast']) > 50 else movie['cast'],
            'similarity': f"{score * 100:.1f}%"
        })

    return results, corrected


@app.route('/')
def home():
    # Get top rated movies for homepage
    top_movies = df.nlargest(6, 'rating')[['title', 'year', 'director']].to_dict('records')
    return render_template('index.html', popular=top_movies)


@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form.get('movie', '')
    if not movie:
        return jsonify({'error': 'Enter a movie name'})

    recommendations, corrected = get_recommendations(movie)

    if not recommendations:
        # Get suggestions
        suggestions = [m[0] for m in process.extract(movie, df['title'].tolist(), limit=3) if m[1] > 50]
        return jsonify({'error': f"'{movie}' not found", 'suggestions': suggestions})

    return jsonify({
        'success': True,
        'corrected': corrected if corrected.lower() != movie.lower() else None,
        'original': movie,
        'movies': recommendations
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)