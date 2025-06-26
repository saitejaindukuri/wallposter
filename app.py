from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import os
import random
import hashlib
import sqlite3
from surprise import Dataset, Reader, SVD

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATA_DIR = 'data'
DB_PATH = os.path.join(DATA_DIR, 'wallposter.db')
TRAINING_SET_DIR = os.path.join(DATA_DIR, 'training_set')
MOVIE_TITLES_FILE = os.path.join(DATA_DIR, 'movie_titles.txt')

# Ensure DB file exists and tables created
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                password TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                movie_title TEXT
            )
        ''')
init_db()

def get_db():
    return sqlite3.connect(DB_PATH)

def create_user(name, password):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('INSERT INTO users (name, password) VALUES (?, ?)', (name, password))
        conn.commit()
        return cur.lastrowid

def add_to_watchlist_db(user_id, movie_id, movie_title):
    with get_db() as conn:
        conn.execute('INSERT INTO watchlists (user_id, movie_id, movie_title) VALUES (?, ?, ?)', (user_id, movie_id, movie_title))
        conn.commit()

def remove_from_watchlist_db(user_id, movie_id):
    with get_db() as conn:
        conn.execute('DELETE FROM watchlists WHERE user_id = ? AND movie_id = ?', (user_id, movie_id))
        conn.commit()

def get_watchlist_from_db(user_id):
    with get_db() as conn:
        cur = conn.execute('SELECT movie_id, movie_title FROM watchlists WHERE user_id = ?', (user_id,))
        return [{'movie_id': row[0], 'title': row[1]} for row in cur.fetchall()]

# ---------------------------------------------
# RECOMMENDATION SYSTEM SETUP
# ---------------------------------------------
def load_movie_titles():
    df = pd.read_csv(MOVIE_TITLES_FILE, header=None, names=['movie_id', 'year', 'title'], encoding='latin-1', on_bad_lines='skip')
    return df.set_index('movie_id')

def load_ratings(sample_size=100):
    rows = []
    files = sorted(os.listdir(TRAINING_SET_DIR))[:sample_size]
    for file in files:
        with open(os.path.join(TRAINING_SET_DIR, file), 'r') as f:
            movie_id = int(f.readline().strip()[:-1])
            for line in f:
                user_id, rating, date = line.strip().split(',')
                rows.append((int(user_id), movie_id, int(rating)))
    return pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating'])

movie_titles = load_movie_titles()
df = load_ratings()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

def generate_fake_poster():
    return f"https://picsum.photos/200/300?random={random.randint(1, 10000)}"

def generate_fake_description():
    return random.choice([
        "A thrilling journey through time.",
        "An unforgettable tale of love and loss.",
        "Action-packed adventure like no other.",
        "A mystery that keeps you on the edge.",
        "An emotional drama with powerful performances.",
        "Sci-fi spectacle beyond imagination.",
    ])

def generate_fake_genres():
    return random.choice(["Action", "Drama", "Comedy", "Sci-Fi", "Thriller"])

def generate_consistent_name(user_id):
    first_names = ['Alex', 'Sam', 'Jordan', 'Taylor', 'Morgan', 'Jamie', 'Charlie', 'Drew']
    last_names = ['Smith', 'Lee', 'Patel', 'Brown', 'Garcia', 'Jones', 'Davis', 'Wang']
    seed = int(hashlib.sha256(str(user_id).encode()).hexdigest(), 16)
    return f"{first_names[seed % len(first_names)]} {last_names[(seed // len(first_names)) % len(last_names)]}"

def get_top_n(user_id, n=5):
    rated_movies = df[df['user_id'] == user_id]['movie_id'].tolist()
    all_movies = df['movie_id'].unique()
    to_predict = [mid for mid in all_movies if mid not in rated_movies]
    predictions = [algo.predict(user_id, mid) for mid in to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    return [{
        'movie_id': int(pred.iid),
        'title': movie_titles.loc[int(pred.iid)]['title'],
        'poster': generate_fake_poster(),
        'description': generate_fake_description(),
        'genre': generate_fake_genres()
    } for pred in top_n]

def get_trending_now(n=7):
    most_rated = df['movie_id'].value_counts().head(50).index.tolist()
    trending_sample = random.sample(most_rated, min(n, len(most_rated)))
    return [{
        'movie_id': mid,
        'title': movie_titles.loc[mid]['title'],
        'poster': generate_fake_poster(),
        'description': generate_fake_description(),
        'genre': generate_fake_genres()
    } for mid in trending_sample]

def get_random_movies(n=14):
    random_ids = random.sample(list(movie_titles.index), n)
    return [{
        'movie_id': mid,
        'title': movie_titles.loc[mid]['title'],
        'poster': generate_fake_poster()
    } for mid in random_ids]

# ---------------------------------------------
# FLASK ROUTES
# ---------------------------------------------

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    results = movie_titles[movie_titles['title'].str.lower().str.contains(query)]
    formatted_results = [{
        'movie_id': mid,
        'title': row['title'],
        'poster': generate_fake_poster(),
        'description': generate_fake_description(),
        'genre': generate_fake_genres()
    } for mid, row in results.iterrows()]
    return render_template('search_results.html',
                           name=session.get('name'),
                           user_id=session.get('user_id'),
                           results=formatted_results,
                           query=query)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        session['user_id'] = user_id
        session['name'] = generate_consistent_name(user_id)
        return redirect(url_for('recommendations'))
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form['name']
    password = request.form['password']
    user_id = create_user(name, password)
    session['user_id'] = user_id
    session['name'] = name
    return redirect(url_for('select_preferences'))

@app.route('/select_preferences')
def select_preferences():
    user_id = session.get('user_id')
    name = session.get('name')
    if not user_id or not name:
        return redirect(url_for('login'))
    movies = get_random_movies()
    return render_template("select_preferences.html", movies=movies, name=name, uid=user_id, uname=name, new=True)

@app.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    selected = request.form.getlist('selected_movies')
    user_id = session.get('user_id')
    print(f"User {user_id} selected: {selected}")
    return redirect(url_for('recommendations'))

@app.route('/recommendations')
def recommendations():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    name = session.get('name')
    recommendations = get_top_n(user_id)
    trending_now = get_trending_now()
    return render_template('recommendations.html',
                           recommendations=recommendations,
                           trending_now=trending_now,
                           name=name,
                           user_id=user_id)

@app.route('/watchlist')
def watchlist():
    user_id = session.get('user_id')
    name = session.get('name')
    watchlist_movies = get_watchlist_from_db(user_id)
    return render_template('watchlist.html',
                           name=name,
                           user_id=user_id,
                           watchlist=watchlist_movies)

@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    user_id = session.get('user_id')
    data = request.json
    movie_id = data.get('movie_id')
    title = data.get('title')
    add_to_watchlist_db(user_id, movie_id, title)
    return jsonify({'status': 'success'})

@app.route('/remove_from_watchlist', methods=['POST'])
def remove_from_watchlist():
    user_id = session.get('user_id')
    movie_id = int(request.form['movie_id'])
    remove_from_watchlist_db(user_id, movie_id)
    return redirect(url_for('watchlist'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)
