# NBA Lineup Prediction Model  

**Course**: SOFE 4620U - Machine Learning and Data Mining  

**Group ML5 Members**:  
Hamzi Farhat 100831450  
Jason Stuckless 100248154  
Tahmid Chowdhury 100822671  

## Overview
This Python program predicts the fifth home team player (`home_4`) in NBA matchups using historical game data (2007–2015). It leverages **player win rates**, **pairwise synergy scores**, and an **XGBoost classifier** to generate predictions, evaluating both top-1 and top-3 accuracy.

## Key Features
- **Player Win Rates**: Calculates individual player historical win rates.
- **Synergy Scores**: Measures pairwise player performance to create a team synergy feature.
- **XGBoost Model**: Trains a classifier with custom hyperparameters (`max_depth=10`, `n_estimators=200`).
- **Evaluation Metrics**: Reports top-1 and top-3 accuracy, precision, recall, and F1-score.

## How It Works
### Workflow Steps:
1. **Data Loading**:  
   - Reads yearly matchup CSV files (e.g., `matchups-2007.csv`).
   - Filters columns to teams, players, and game outcomes.

2. **Feature Engineering**:  
   - **Player Win Rates**: Aggregates historical win rates for each player.  
   - **Synergy Scores**: Computes average win rates for all player pairs in a lineup.  

3. **Data Preprocessing**:  
   - Encodes categorical features (teams, players) using `LabelEncoder`.  
   - Handles unseen players in test data with an "unknown" category.  

4. **Model Training**:  
   - Uses `XGBClassifier` to predict the fifth home team player (`home_4`).  

5. **Evaluation**:  
   - Evaluates predictions using **top-1** (exact match) and **top-3** (true label in top 3 predictions) metrics.  
   - Saves results to `training_testing_results.csv`.

## Results
The program outputs a CSV file with yearly evaluation metrics:
- **Top-1 Accuracy**: Accuracy of the highest-confidence prediction.  
- **Top-3 Accuracy**: Whether the true player is in the top 3 predictions.  
- Precision, recall, and F1-scores for both modes.

## Requirements
### Libraries
- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

### Input Data Format
CSV files named `matchups-YYYY.csv` with columns:  
- `home_team`, `away_team`, `home_0` to `home_4`, `away_0` to `away_4`, `outcome`.

---

## Usage
1. Install the required libraries using the following command: `pip install numpy pandas scikit-learn xgboost matplotlib`
2. Ensure CSV files are in the data directory.  
3. Run the script: Results are printed to terminal and multiple csv files are created in data directory depending on mode chosen.  Visualizations are created in the visualization_charts directory for the training/testing pairs mode.

Note: The bulk of this readme file was generated with prompts provided by Jason Stuckless using DeepSeek.
   ```bash
