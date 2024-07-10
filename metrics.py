import csv
import numpy as np

def precision_at_k(session_history, k=10):
    precisions = []
    for session in session_history:
        relevant_items = [movie for movie in session['picked_movies'] if movie['rating'] >= 7]
        precisions.append(len(relevant_items) / k)
    return np.mean(precisions)

def recall_at_k(session_history, k=10):
    recalls = []
    for session in session_history:
        relevant_items = [movie for movie in session['picked_movies'] if movie['rating'] >= 7]
        all_relevant_items = [movie for movie in session['picked_movies']]
        recalls.append(len(relevant_items) / len(all_relevant_items) if all_relevant_items else 0)
    return np.mean(recalls)

def mean_average_precision(session_history):
    avg_precisions = []
    for session in session_history:
        relevant_items = [movie for movie in session['picked_movies'] if movie['rating'] >= 7]
        if not relevant_items:
            continue
        precision_sum = 0
        for i, movie in enumerate(relevant_items, 1):
            precision_sum += len(relevant_items[:i]) / i
        avg_precisions.append(precision_sum / len(relevant_items))
    return np.mean(avg_precisions)

def mean_reciprocal_rank(session_history):
    mrrs = []
    for session in session_history:
        relevant_items = [i for i, movie in enumerate(session['picked_movies'], 1) if movie['rating'] >= 7]
        if not relevant_items:
            continue
        mrrs.append(1 / relevant_items[0])
    return np.mean(mrrs)

def rmse(session_history):
    errors = []
    for session in session_history:
        for movie in session['picked_movies']:
            predicted_rating = movie['rating']
            actual_rating = movie['watch_percentage'] / 10  # Assuming watch_percentage as a proxy for actual rating
            errors.append((predicted_rating - actual_rating) ** 2)
    return np.sqrt(np.mean(errors))

def ndcg(session_history, k=10):
    ndcgs = []
    for session in session_history:
        dcg = 0
        idcg = 0
        relevant_items = [movie for movie in session['picked_movies'] if movie['rating'] >= 7]
        for i, movie in enumerate(relevant_items[:k], 1):
            dcg += (2 ** movie['rating'] - 1) / np.log2(i + 1)
            idcg += (2 ** 10 - 1) / np.log2(i + 1)
        if idcg == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs)

def load_session_history_from_csv(file_name):
    session_history = []
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        current_session = None
        for row in reader:
            session_num = int(row['session'])
            if current_session is None or current_session['session'] != session_num:
                if current_session:
                    session_history.append(current_session)
                current_session = {
                    'session': session_num,
                    'initial_movie': row['initial_movie'],
                    'picked_movies': []
                }
            current_session['picked_movies'].append({
                'picked_movie': row['picked_movie'],
                'genre': row['genre'],
                'rating': int(row['rating']),
                'timestamp': row['timestamp'],
                'watch_percentage': int(row['watch_percentage']),
                'position': int(row['position'])
            })
        if current_session:
            session_history.append(current_session)
    return session_history

def calculate_metrics(session_history):
    precision = precision_at_k(session_history, k=10)
    recall = recall_at_k(session_history, k=10)
    map_ = mean_average_precision(session_history)
    mrr = mean_reciprocal_rank(session_history)
    rmse_val = rmse(session_history)
    ndcg_val = ndcg(session_history, k=10)

    print(f"Precision@10: {precision}")
    print(f"Recall@10: {recall}")
    print(f"MAP: {map_}")
    print(f"MRR: {mrr}")
    print(f"RMSE: {rmse_val}")
    print(f"NDCG@10: {ndcg_val}")

def log_additional_info(session_history):
    total_movies_watched = sum(len(session['picked_movies']) for session in session_history)
    full_watch_sessions = sum(1 for session in session_history if len(session['picked_movies']) == 3)
    total_watch_time = sum(movie['watch_percentage'] for session in session_history for movie in session['picked_movies']) / 100

    print(f"Total movies watched: {total_movies_watched}")
    print(f"Sessions with 3/3 movies watched: {full_watch_sessions}")
    print(f"Total watch time (hours): {total_watch_time}")

def main():
    session_history = load_session_history_from_csv('agent_session_history.csv')

    calculate_metrics(session_history)
    log_additional_info(session_history)

if __name__ == "__main__":
    main()
