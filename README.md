# NBA Lineup Prediction Model  

**Course**: SOFE 4620U - Machine Learning and Data Mining  

**Group ML5 Members**:  
Hamzi Farhat 100831450  
Jason Stuckless 100248154  
Tahmid Chowdhury 100822671  

## Overview
This Python program predicts the fifth home team player (`home_4`) in NBA matchups using historical game data (2007–2015). It provides two modes of operation:

1. **Test Data Mode**: This mode evaluates the model using a predefined test dataset (`NBA_test.csv`). It processes missing labels, applies model predictions, and computes accuracy metrics.
2. **Year Pair Mode**: This mode trains the model on historical year matchups and evaluates its performance on subsequent years (e.g., training on 2007 data and testing on 2008). It provides insights into the model's generalization over time.

Both modes utilize **player win rates**, **pairwise synergy scores**, and an **XGBoost classifier** to generate predictions, evaluating both top-1 and top-3 accuracy.

## Key Features
- **Player Win Rates**: Calculates individual player historical win rates.
- **Synergy Scores**: Measures pairwise player performance to create a team synergy feature.
- **Player Frequency Analysis**: Computes how often players appear in past lineups.
- **Team Win Rates**: Tracks team-wide success rates to refine predictions.
- **GPU Acceleration**: Utilizes CUDA-enabled XGBoost for faster training if available.
- **Comprehensive Feature Engineering**: Includes player synergy, win rate differentials, and frequency-based metrics.
- **Data Validation & Preprocessing**: Checks for mismatches in training/test labels and encodes categorical features.
- **XGBoost Model**: Trains a classifier with custom hyperparameters (`max_depth=8`, `n_estimators=150`, `learning_rate=0.1`).
- **Evaluation Metrics**: Reports top-1 and top-3 accuracy, precision, recall, and F1-score.
- **Visualization**: Generates plots comparing model performance over different years.

## How It Works
### Workflow Steps:
1. **Data Loading**:  
   - Reads yearly matchup CSV files (e.g., `matchups-2007.csv`).
   - Handles missing or incorrectly formatted data.
   
2. **Feature Engineering**:  
   - **Player Win Rates**: Aggregates historical win rates for each player.  
   - **Synergy Scores**: Computes average win rates for all player pairs in a lineup.  
   - **Player Frequency Features**: Tracks how often each player appears in matchups.  
   - **Team Win Rates**: Adds a team-level win rate feature.  
   - **Synergy Metrics**: Measures team composition using synergy-based features.
   
3. **Data Preprocessing**:  
   - Encodes categorical features (teams, players) using `LabelEncoder`.  
   - Normalizes numerical features using `StandardScaler`.  
   - Handles unseen players in test data with default or estimated values.  
   
4. **Model Training**:  
   - Uses `XGBClassifier` with GPU acceleration (if available) to predict `home_4`.  
   - Implements optimized hyperparameters for improved performance.
   
5. **Evaluation & Visualization**:  
   - Evaluates predictions using **top-1** (exact match) and **top-3** (true label in top 3 predictions) metrics.  
   - Generates accuracy, precision, recall, and F1-score comparisons.  
   - Saves performance results to `data/results.csv` and plots in `visualization_charts/`.

## Results
The program outputs evaluation metrics across different years:
- **Top-1 Accuracy**: Measures correctness of the highest-confidence prediction.  
- **Top-3 Accuracy**: Determines if the correct player is among the top 3 predictions.  
- **Precision, Recall, and F1-score**: Additional evaluation metrics for deeper analysis.
- **Visualization Charts**: Graphical comparison of model performance across seasons.

## Requirements
### Libraries
- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`

### Input Data Format
CSV files named `matchups-YYYY.csv` with columns:  
- `home_team`, `away_team`, `home_0` to `home_4`, `away_0` to `away_4`, `outcome`.

## Usage
1. Install the required libraries using the following command:
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib
   ```
2. Ensure CSV files are in the `data/` directory.  
3. Run the script:
   ```bash
   python nba_lineup_prediction.py
   ```
4. Select one of the execution modes:
   - **Test Data Mode**: Uses `NBA_test.csv` for evaluation.
   - **Year Pair Mode**: Trains and tests across historical year matchups.
5. Results are printed to the terminal and saved to `data/results.csv`.  
6. Visualizations are saved in `visualization_charts/` for analysis.

---

### Notes
- The README has been updated to reflect new features and model improvements.
- The script will automatically detect GPU availability and adjust training settings.
- If test data labels are missing, warnings will be displayed.
