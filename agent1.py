import numpy as np
import random
from datetime import datetime
import requests
import json
import csv
import time


class MovieAgent:
    def __init__(self, config):
        self.preferences = config['initial_preferences']
        self.q_table = {genre: np.zeros(10) for genre in self.preferences}  # Q-table for each genre
        self.alpha = config['alpha']  # Learning rate
        self.gamma = config['gamma']  # Discount factor
        self.no_pick_chance = config['no_pick_chance']  # Initial chance of not picking a movie
        self.explore_chance = config['explore_chance']  # Initial chance of exploring a new genre
        self.base_url = config['base_url']  # URL of the running Flask app
        self.session_history = []  # Log of sessions
        self.genre_history = []  # Track genres watched to simulate boredom
        self.recently_watched = []  # Track recently watched movies to avoid immediate repetition

        # Initialize CSV logging
        self.csv_file = 'agent_session_history.csv'
        with open(self.csv_file, 'w', newline='') as csvfile:
            fieldnames = ['session', 'initial_movie', 'picked_movie', 'genre', 'rating', 'timestamp',
                          'watch_percentage', 'position']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def update_preferences(self, genre, rating):
        if genre not in self.q_table:
            self.q_table[genre] = np.zeros(10)
        current_q = self.q_table[genre][rating - 1]
        max_future_q = max(self.q_table[genre])
        new_q = current_q + self.alpha * (rating + self.gamma * max_future_q - current_q)
        self.q_table[genre][rating - 1] = new_q
        self.preferences[genre] = np.mean(self.q_table[genre])

    def pick_movie(self, recommended_movies):
        valid_movies = [m for m in recommended_movies if m['title'] not in self.recently_watched]

        if len(valid_movies) == 0:
            print("No valid movies to pick from after filtering.")
            if len(recommended_movies) > 0:
                print("Considering recently watched movies as well.")
                valid_movies = recommended_movies

        if len(valid_movies) == 0:
            print("No valid movies to pick from even after considering recently watched.")
            return None

        if random.random() < self.no_pick_chance:
            print("No movie picked due to no_pick_chance.")
            return None

        if random.random() < self.explore_chance or self.is_bored():
            # Explore a new genre
            picked_movie = random.choice(valid_movies)
            genre = picked_movie['genre'].split(', ')[0]  # Ensure only the first genre is considered
            print(f"Exploring a new genre: {genre}")
        else:
            for movie in valid_movies:
                movie['genre'] = movie['genre'].split(', ')[0]  # Ensure only the first genre is considered
                if 'position' not in movie:
                    movie['position'] = 0  # Set default position if not present
            sorted_movies = sorted(valid_movies, key=lambda x: (self.preferences.get(x['genre'], 0), -x['position']),
                                   reverse=True)
            picked_movie = sorted_movies[0]

        rating = self.get_rating(picked_movie['genre'])
        watch_percentage = self.calculate_watch_percentage(rating, picked_movie['position'])
        self.update_preferences(picked_movie['genre'], rating)
        timestamp = datetime.now()
        self.genre_history.append(picked_movie['genre'])
        self.update_probabilities(rating)
        self.recently_watched.append(picked_movie['title'])  # Add the picked movie to recently watched list
        if len(self.recently_watched) > 10:  # Limit the memory size to 10 movies
            self.recently_watched.pop(0)

        return {
            'picked_movie': picked_movie['title'],
            'genre': picked_movie['genre'],
            'rating': rating,
            'timestamp': timestamp,
            'watch_percentage': watch_percentage,
            'position': picked_movie['position']
        }

    def get_rating(self, genre):
        if genre not in self.q_table:
            self.q_table[genre] = np.zeros(10)
        q_values = self.q_table[genre]
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        return np.random.choice(np.arange(1, 11), p=probabilities)

    def calculate_watch_percentage(self, rating, position):
        # Higher ratings and top positions in the recommendation list result in higher watch percentages
        if rating >= 8:
            return random.randint(80, 100)
        elif rating >= 5:
            return random.randint(50, 80)
        else:
            return random.randint(10, 50)

    def update_probabilities(self, rating):
        # Adjust no_pick_chance and explore_chance based on the rating of the last watched movie
        if rating >= 8:
            self.no_pick_chance = max(0.01, self.no_pick_chance - 0.01)
            self.explore_chance = max(0.05, self.explore_chance - 0.01)
        elif rating < 5:
            self.no_pick_chance = min(0.2, self.no_pick_chance + 0.01)
            self.explore_chance = min(0.3, self.explore_chance + 0.01)

    def get_recommendations(self, movie_title):
        response = requests.post(f'{self.base_url}/recommend', data={'name': movie_title})
        if response.status_code == 200:
            recommendations = response.json().get('recommendations', [])
            for i, rec in enumerate(recommendations):
                rec['position'] = i + 1  # Add position to each recommendation
            return recommendations
        return []

    def is_bored(self):
        if len(self.genre_history) < 3:
            return False
        return all(genre == self.genre_history[-1] for genre in self.genre_history[-3:])

    def simulate_session(self, initial_movie, session_num):
        session_log = {
            'session': session_num,
            'initial_movie': initial_movie,
            'picked_movies': [],
            'preferences': self.preferences.copy()
        }
        recommended_movies = self.get_recommendations(initial_movie)
        watched_movie_titles = set()
        for _ in range(3):  # Simulate up to 3 movie selections per session
            if recommended_movies:
                decision = self.pick_movie(recommended_movies)
                if decision and 'picked_movie' in decision and decision['picked_movie'] not in watched_movie_titles:
                    watched_movie_titles.add(decision['picked_movie'])
                    session_log['picked_movies'].append(decision)
                    self.log_to_csv(session_num, initial_movie, decision)
                    print(
                        f"Picked movie: {decision['picked_movie']}, Genre: {decision['genre']}, Rating: {decision['rating']}, Watch Percentage: {decision['watch_percentage']}%, Position: {decision['position']}, Timestamp: {decision['timestamp']}")
                    recommended_movies = self.get_recommendations(
                        decision['picked_movie'])  # Get new recommendations based on the picked movie
                else:
                    print("No movie picked this session")
                    break
            else:
                print("No recommendations found for this movie")
                break
        self.session_history.append(session_log)
        print(f"Updated preferences: {self.preferences}")
        time.sleep(1)  # Adding delay of 0.1 seconds between sessions

    def log_to_csv(self, session_num, initial_movie, decision):
        with open(self.csv_file, 'a', newline='') as csvfile:
            fieldnames = ['session', 'initial_movie', 'picked_movie', 'genre', 'rating', 'timestamp',
                          'watch_percentage', 'position']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'session': session_num,
                'initial_movie': initial_movie,
                'picked_movie': decision['picked_movie'],
                'genre': decision['genre'],
                'rating': decision['rating'],
                'timestamp': decision['timestamp'],
                'watch_percentage': decision['watch_percentage'],
                'position': decision['position']
            })

    def run_simulation(self, num_sessions):
        for session_num in range(num_sessions):
            print(f"\n--- Session {session_num + 1} ---")
            initial_movie = random.choice(['Inception', 'The Dark Knight', 'Pulp Fiction', 'The Matrix', 'Fight Club'])
            self.simulate_session(initial_movie, session_num + 1)


def main():
    config = {
        'initial_preferences': {'Action': 0, 'Comedy': 0, 'Drama': 0, 'Horror': 0},
        'alpha': 0.1,
        'gamma': 0.9,
        'no_pick_chance': 0.1,
        'explore_chance': 0.4,  # 20% chance to explore a new genre
        'base_url': 'http://127.0.0.1:5000'
    }
    agent = MovieAgent(config)
    agent.run_simulation(num_sessions=100)  # Simulate 1000 sessions

    # Save session history to a file for analysis
    with open('session_history.json', 'w') as f:
        json.dump(agent.session_history, f, indent=4, default=str)


if __name__ == "__main__":
    main()
