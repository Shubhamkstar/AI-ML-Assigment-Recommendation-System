# config.py

# Configuration settings for the MovieAgent
config = {
    'initial_preferences': {'Action': 5, 'Comedy': 5, 'Drama': 5, 'Horror': 5},
    'alpha': 0.1,  # Learning rate
    'gamma': 0.9,  # Discount factor
    'no_pick_chance': 0.2,  # Probability of not picking a movie
    'random_watcher': False  # If True, the agent picks movies randomly

}
