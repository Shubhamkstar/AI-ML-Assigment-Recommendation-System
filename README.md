# AI-ML-Assigment-Recommendation-System
Certainly! Here is the complete README file in a single block for easy copy-pasting:

```markdown
# Self-Adaptive Movie Recommendation System

## Project Overview

This project aims to create a self-adaptive movie recommendation system that dynamically adjusts its parameters to improve recommendation accuracy over time. The system is built upon an existing baseline recommendation engine and incorporates a human-like agent to simulate realistic movie-watching behavior. The self-adaptive mechanism is implemented using the MAPE-K framework, which continuously monitors system performance, analyzes interaction data, plans parameter adjustments, and executes these adaptations in real-time.

## Repository Structure

- `main.py`: The baseline movie recommendation system.
- `agent1.py`: The human-like agent that interacts with the recommendation system.
- `mape_k.py`: The self-adaptive mechanism that tunes the system's parameters.
- `metrics.py`: Script to evaluate the performance of the system.
- `params.csv`: Configuration file for the self-adaptive mechanism.
- `agent_session_history.csv`: Log file for agent sessions.
- `session_history.json`: Log file for detailed session history.

## Running Instructions

### Step 1: Set Up the Baseline Recommendation System

1. Ensure you have all the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn flask requests
   ```

2. Load the dataset into the `main.py` script:
   ```python
   data = pd.read_csv('path/to/main_data.csv')
   ```

3. Run the baseline recommendation system:
   ```bash
   python main.py
   ```
   The system will start a Flask server to handle recommendation requests.

### Step 2: Configure the Human-Like Agent

1. Update the `params.csv` file with the initial configuration parameters:
   ```csv
   parameter,value
   alpha,0.1
   gamma,0.9
   num_recommendations,10
   imdb_rating_threshold,5
   diversity,2
   ```

2. Set the agent parameters in the `agent1.py` script:
   ```python
   config = {
       'initial_preferences': {'Action': 0, 'Comedy': 0, 'Drama': 0, 'Horror': 0},
       'alpha': 0.1,
       'gamma': 0.9,
       'no_pick_chance': 0.1,
       'explore_chance': 0.4,
       'base_url': 'http://127.0.0.1:5000'
   }
   ```

3. Ensure the `time.sleep(1)` statement in `agent1.py` is set to a suitable delay between sessions:
   ```python
   time.sleep(1)  # Adding delay of 1 second between sessions
   ```

4. Specify the number of sessions to run:
   ```python
   agent.run_simulation(num_sessions=100)
   ```

5. Clear the `agent_session_history.csv` and `session_history.json` files to avoid conflicts with previous runs:
   ```bash
   > agent_session_history.csv
   > session_history.json
   ```

6. Run the agent to simulate user interactions:
   ```bash
   python agent1.py
   ```

### Step 3: Run the Self-Adaptive Mechanism

1. Start the MAPE-K loop to monitor and adapt the system:
   ```bash
   python mape_k.py
   ```

### Step 4: Evaluate the System's Performance

1. Use `metrics.py` to evaluate the performance of the recommendation system:
   ```bash
   python metrics.py
   ```

2. The `metrics.py` script evaluates the performance based on `agent_session_history.csv`:
   - Save `agent_session_history.csv` and `session_history.json` manually to avoid confusion and ensure you have the correct files for evaluation.

### Notes

- The baseline recommendation system must be running while the agent and MAPE-K loop are in operation.
- Adjust the parameters in `params.csv` and `agent1.py` as needed to experiment with different configurations.
- The `metrics.py` script will output key performance metrics such as Precision@10, Recall@10, MAP, MRR, RMSE, and NDCG@10.

## Conclusion

This project demonstrates a novel approach to creating a self-adaptive movie recommendation system using a human-like agent and the MAPE-K framework. By continuously monitoring and adapting the system's parameters, we achieve significant improvements in recommendation accuracy and user engagement. Future work will focus on further refining the adaptation logic and exploring additional parameters to enhance performance.

## License

This project is licensed under the MIT License.
