from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import secrets

app = Flask(__name__)
# Use a stable secret key in production so sessions work across Gunicorn workers.
# Set SECRET_KEY in your environment; falls back to a random key for local runs.
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')

# Neural CF Model Definition (same as notebook)
class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NeuralCF, self).__init__()

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.BatchNorm1d(layer_size))
            mlp_modules.append(nn.Dropout(dropout))
            input_size = layer_size

        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Final prediction layer
        self.prediction = nn.Linear(embedding_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.constant_(self.prediction.bias, 0)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_vector = gmf_user_emb * gmf_item_emb

        # MLP path
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # Concatenate and predict
        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(combined_vector)
        return prediction.squeeze()


# Load data
print("Loading data...")
BASE_DIR = Path(__file__).resolve().parent
movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
             ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv(BASE_DIR / 'data/u.item', sep='|', names=movie_cols, encoding='latin-1')

ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(BASE_DIR / 'data/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# Load user data for demographic recommendations
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv(BASE_DIR / 'data/u.user', sep='|', names=user_cols, encoding='latin-1')

# Get dataset stats
n_users = ratings['user_id'].max()
n_items = ratings['item_id'].max()

print(f"Loaded {len(movies)} movies, {n_users} users, {len(ratings)} ratings")

# Load model with exact parameters from notebook
print("Loading model...")
model = NeuralCF(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=128,  # From best params
    mlp_layers=[256, 128, 64, 32],  # From best params
    dropout=0.3  # From best params
)

# Load trained weights
try:
    model.load_state_dict(torch.load(BASE_DIR / 'best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Get genre for each movie
def get_genres(movie_row):
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = [g for g in genre_cols if movie_row[g] == 1]
    return ', '.join(genres) if genres else 'Unknown'

movies['genres'] = movies.apply(get_genres, axis=1)

# Get popular movies for cold start
def get_popular_movies(n=100):
    """Get most popular movies based on rating count and average"""
    movie_stats = ratings.groupby('item_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['item_id', 'count', 'avg_rating']
    movie_stats = movie_stats[movie_stats['count'] >= 50]
    movie_stats['score'] = movie_stats['avg_rating'] * 0.7 + \
                           (movie_stats['count'] / movie_stats['count'].max()) * 5 * 0.3
    return movie_stats.sort_values('score', ascending=False).head(n)['item_id'].tolist()

popular_movie_ids = get_popular_movies()

def get_demographic_recommendations(age, gender, occupation, top_k=10):
    """Get recommendations based on demographic similarity"""
    # Find similar users
    similar_users = users[
        (users['age'].between(age-10, age+10)) & 
        (users['gender'] == gender) &
        (users['occupation'] == occupation)
    ]['user_id'].values - 1  # Adjust for 0-indexing
    
    if len(similar_users) == 0:
        # Fallback to popular movies
        return popular_movie_ids[:top_k]
    
    # Get highly rated movies from similar users
    similar_user_ratings = ratings[
        (ratings['user_id'].isin(similar_users)) &
        (ratings['rating'] >= 4)
    ]
    
    # Calculate popularity among similar users
    movie_stats = similar_user_ratings.groupby('item_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['item_id', 'count', 'avg_rating']
    movie_stats['score'] = movie_stats['avg_rating'] * np.log1p(movie_stats['count'])
    
    top_movies = movie_stats.sort_values('score', ascending=False).head(top_k)['item_id'].tolist()
    return top_movies

def get_recommendations_for_user(rated_items, top_k=10):
    """
    Get recommendations based on user's ratings
    rated_items: dict of {item_id: rating}
    """
    if len(rated_items) == 0:
        # Cold start: return popular movies
        candidate_ids = popular_movie_ids[:top_k]
        return candidate_ids

    # Convert keys to integers (Flask session serialization converts them to strings)
    rated_items = {int(k): v for k, v in rated_items.items()}

    # Use average user embedding as proxy for new user
    # Weight by ratings to create personalized profile
    model.eval()

    # Get candidate items (exclude already rated)
    all_items = set(range(1, n_items + 1))
    rated_item_ids = set(rated_items.keys())
    candidate_items = list(all_items - rated_item_ids)

    if len(candidate_items) == 0:
        return []

    # Create a synthetic user profile based on rated items
    # Use the mean of item embeddings weighted by ratings
    with torch.no_grad():
        rated_item_tensors = torch.LongTensor([item_id - 1 for item_id in rated_items.keys()]).to(device)
        rated_ratings = np.array([rated_items[item_id] for item_id in rated_items.keys()])

        # Get item embeddings
        item_embs = model.gmf_item_embedding(rated_item_tensors)

        # Weight by ratings (normalize)
        weights = torch.FloatTensor(rated_ratings / 5.0).unsqueeze(1).to(device)
        user_profile = (item_embs * weights).mean(dim=0, keepdim=True)

        # Score all candidate items
        candidate_tensors = torch.LongTensor([item_id - 1 for item_id in candidate_items]).to(device)
        candidate_embs = model.gmf_item_embedding(candidate_tensors)

        # Compute similarity scores
        scores = (user_profile * candidate_embs).sum(dim=1).cpu().numpy()

    # Get top-K items
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_items = [candidate_items[i] for i in top_indices]

    return top_items


@app.route('/')
def index():
    # Check if demographics are collected
    if 'demographics' not in session:
        return redirect(url_for('demographics'))
    
    # Initialize session if needed
    if 'user_ratings' not in session:
        session['user_ratings'] = {}
        session['rated_count'] = 0
    
    # Initialize shown_movies tracking
    if 'shown_movies' not in session:
        session['shown_movies'] = []

    # Get initial 5 movies
    if session['rated_count'] == 0:
        # Use demographic-based recommendations for cold start
        demo = session['demographics']
        initial_movies = get_demographic_recommendations(
            age=int(demo['age']),
            gender=demo['gender'],
            occupation=demo['occupation'],
            top_k=20
        )
    else:
        # Get recommendations (request more to account for filtering)
        initial_movies = get_recommendations_for_user(session['user_ratings'], top_k=20)
    
    # Filter out already shown movies and take first 5
    shown_set = set(session['shown_movies'])
    unique_movies = [m for m in initial_movies if m not in shown_set][:5]
    
    # Add to shown_movies list
    session['shown_movies'].extend(unique_movies)
    session.modified = True

    # Get movie details
    movie_data = []
    for movie_id in unique_movies:
        movie_info = movies[movies['item_id'] == movie_id].iloc[0]
        movie_data.append({
            'id': int(movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'year': movie_info['release_date'].split('-')[-1] if pd.notna(movie_info['release_date']) else 'Unknown'
        })

    return render_template('index.html', movies=movie_data, rated_count=session['rated_count'])


@app.route('/rate', methods=['POST'])
def rate_movie():
    data = request.json
    movie_id = data.get('movie_id')
    rating = data.get('rating')

    if not movie_id or not rating:
        return jsonify({'error': 'Missing data'}), 400

    # Store rating
    if 'user_ratings' not in session:
        session['user_ratings'] = {}
    
    # Initialize shown_movies tracking
    if 'shown_movies' not in session:
        session['shown_movies'] = []

    # Store as string key for JSON serialization compatibility
    session['user_ratings'][str(movie_id)] = int(rating)
    session['rated_count'] = len(session['user_ratings'])
    session.modified = True

    # Get next recommendation (request more to account for filtering)
    recommendations = get_recommendations_for_user(session['user_ratings'], top_k=20)
    
    # Filter out already shown movies
    shown_set = set(session['shown_movies'])
    unique_recommendations = [m for m in recommendations if m not in shown_set]

    if len(unique_recommendations) == 0:
        return jsonify({'no_more_movies': True})

    next_movie_id = unique_recommendations[0]
    
    # Add to shown_movies list
    session['shown_movies'].append(next_movie_id)
    session.modified = True
    movie_info = movies[movies['item_id'] == next_movie_id].iloc[0]

    next_movie = {
        'id': int(next_movie_id),
        'title': movie_info['title'],
        'genres': movie_info['genres'],
        'year': movie_info['release_date'].split('-')[-1] if pd.notna(movie_info['release_date']) else 'Unknown'
    }

    return jsonify({
        'success': True,
        'next_movie': next_movie,
        'rated_count': session['rated_count']
    })


@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    # Re-initialize empty tracking
    session['user_ratings'] = {}
    session['rated_count'] = 0
    session['shown_movies'] = []
    return jsonify({'success': True})


@app.route('/stats')
def stats():
    if 'user_ratings' not in session:
        return jsonify({'rated_count': 0})

    return jsonify({
        'rated_count': len(session['user_ratings']),
        'average_rating': np.mean(list(session['user_ratings'].values())) if session['user_ratings'] else 0
    })


@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        
        if age and gender and occupation:
            session['demographics'] = {
                'age': age,
                'gender': gender,
                'occupation': occupation
            }
            return redirect(url_for('index'))
    
    return render_template('demographics.html')


if __name__ == '__main__':
    print("\n" + "="*80)
    print("NEURAL COLLABORATIVE FILTERING - INTERACTIVE RECOMMENDER")
    print("="*80)
    print(f"Model: Neural CF with embeddings loaded from best_model.pth")
    print(f"Device: {device}")
    print(f"Dataset: {len(movies)} movies, {n_users} users")
    print("="*80)
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser\n")

    # When running directly (development), respect PORT env var if set
    import os
    port = int(os.environ.get('PORT', '8080'))
    app.run(debug=True, host='0.0.0.0', port=port)
