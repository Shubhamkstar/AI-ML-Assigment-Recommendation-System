import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import csv
import datetime
import json

# Load the dataset
data = pd.read_csv('/home/shubhamkulkarni/PycharmProjects/RecommSys_Assignment/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis/main_data.csv')

# Creating a count matrix and a similarity score matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])
similarity = cosine_similarity(count_matrix)

# Load the NLP model and TF-IDF vectorizer from disk
filename = '/home/shubhamkulkarni/PycharmProjects/RecommSys_Assignment/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis/nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('/home/shubhamkulkarni/PycharmProjects/RecommSys_Assignment/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis/tranform.pkl', 'rb'))

def rcmd(m, num_recommendations=10, imdb_rating_threshold=0.0, diversity=0):
    m = m.lower()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:num_recommendations + 1]  # Excluding first item since it is the requested movie itself
        l = []
        unique_genres = set()
        for i in range(len(lst)):
            a = lst[i][0]
            movie_info = data.iloc[a].to_dict()
            # Ensure required fields are included
            movie_info['title'] = movie_info['movie_title']
            movie_info['genre'] = movie_info['genres'].split(', ')[0] if 'genres' in movie_info else 'Unknown'
            movie_info['vote_average'] = movie_info.get('vote_average', 0)
            movie_info['position'] = i + 1  # Add position to each recommendation

            if movie_info['vote_average'] >= imdb_rating_threshold and (diversity == 0 or movie_info['genre'] not in unique_genres):
                l.append(movie_info)
                unique_genres.add(movie_info['genre'])
                if len(unique_genres) >= diversity:
                    break
        return l

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    movie = request.form['name']
    num_recommendations = int(request.form.get('num_recommendations', 10))  # Read from request
    imdb_rating_threshold = float(request.form.get('imdb_rating_threshold', 0.0))  # Read from request
    diversity = int(request.form.get('diversity', 0))  # Read from request
    rc = rcmd(movie, num_recommendations, imdb_rating_threshold, diversity)
    if isinstance(rc, str):
        return jsonify({"error": rc})
    else:
        return jsonify({"recommendations": rc})

if __name__ == '__main__':
    app.run(debug=True)
