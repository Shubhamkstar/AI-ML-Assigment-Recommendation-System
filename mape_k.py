import csv
import time
import json
import random
import numpy as np


class MAPEK:
    def __init__(self, config_file, params_file):
        self.config_file = config_file
        self.params_file = params_file
        self.logs = self.load_logs()
        self.last_adaptation_time = time.time()

    def load_logs(self):
        logs = []
        try:
            with open('agent_session_history.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    logs.append(row)
        except FileNotFoundError:
            pass
        return logs

    def read_params(self):
        params = {}
        with open(self.params_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                params[row['parameter']] = row['value']
        return params

    def write_params(self, params):
        with open(self.params_file, 'w', newline='') as csvfile:
            fieldnames = ['parameter', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in params.items():
                writer.writerow({'parameter': key, 'value': value})

    def analyze(self, logs):
        if len(logs) < 5:
            print("Not enough logs to analyze.")
            return False

        recent_logs = logs[-5:]
        total_movies_watched = sum([1 for log in recent_logs if log['picked_movie']])
        avg_watch_percentage = sum(
            [float(log['watch_percentage']) for log in recent_logs if log['watch_percentage']]) / 5
        avg_movies_per_session = total_movies_watched / 5
        avg_rating = np.mean([int(log['rating']) for log in recent_logs if log['rating']])
        no_movie_picked_sessions = sum([1 for log in recent_logs if log['picked_movie'] == ''])

        print(
            f"Last 5 sessions: Total movies watched: {total_movies_watched}, Avg movies per session: {avg_movies_per_session}, Avg watch percentage: {avg_watch_percentage}, Avg rating: {avg_rating}, No movie picked sessions: {no_movie_picked_sessions}")

        params = self.read_params()
        params['avg_watch_percentage'] = str(avg_watch_percentage)
        params['avg_rating'] = str(avg_rating)
        params['avg_movies_per_session'] = str(avg_movies_per_session)
        params['no_movie_picked_sessions'] = str(no_movie_picked_sessions)
        self.write_params(params)

        if avg_movies_per_session < 3 or no_movie_picked_sessions > 2:
            print("Adaptation needed: Low movies per session or too many no movie picked sessions.")
            return True
        if avg_watch_percentage < 50:
            print("Adaptation needed: Low watch percentage.")
            return True
        if avg_rating < 7:
            print("Adaptation needed: Low average rating.")
            return True

        return False

    def plan(self):
        params = self.read_params()

        # Adjust number of recommendations
        if float(params['avg_watch_percentage']) < 60:
            params['num_recommendations'] = str(min(int(params['num_recommendations']) + 1, 20))
        else:
            params['num_recommendations'] = str(max(int(params['num_recommendations']) - 1, 5))

        # Adjust IMDb rating threshold
        if float(params['avg_rating']) < 7:
            new_imdb_rating_threshold = min(float(params['imdb_rating_threshold']) + 1, 10.0)
            print(f"Adjusting IMDb rating threshold up to: {new_imdb_rating_threshold}")
            params['imdb_rating_threshold'] = str(new_imdb_rating_threshold)
        else:
            new_imdb_rating_threshold = max(float(params['imdb_rating_threshold']) - 0.01, 5.0)
            print(f"Adjusting IMDb rating threshold down to: {new_imdb_rating_threshold}")
            params['imdb_rating_threshold'] = str(new_imdb_rating_threshold)

        # Adjust diversity
        if float(params['avg_movies_per_session']) < 3 or int(params['no_movie_picked_sessions']) > 2:
            params['diversity'] = str(min(int(params['diversity']) + 1, 5))
        else:
            params['diversity'] = str(max(int(params['diversity']) - 1, 1))

        print(f"Planned new parameters: {params}")
        return params

    def execute(self, new_params):
        self.write_params(new_params)
        print(f"Adaptation executed with new parameters: {new_params}")

    def run(self):
        while True:
            new_logs = self.load_logs()
            if len(new_logs) > len(self.logs):
                self.logs = new_logs
                if self.analyze(self.logs):
                    new_params = self.plan()
                    self.execute(new_params)
                else:
                    print("No adaptation needed.")
            else:
                print("No new logs to process.")
            time.sleep(1)


if __name__ == "__main__":
    mape = MAPEK('config.csv', 'params.csv')
    mape.run()
