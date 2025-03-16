import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt

# Set environment variables for GPU usage if available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['OMP_NUM_THREADS'] = '4'  # Control CPU thread usage

# List of matchup file years
train_test_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

# Timer context manager for measuring performance
@contextmanager
def timer(name):
    """Context manager for timing code execution."""
    start_time = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} completed in {end - start_time:.2f} seconds")

# Load CSV file into a DataFrame
def load_data(file):
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        # Try alternate filename format
        file_parts = file.split('-')
        alternate_file = f"data/matchups{file_parts[1].split('.')[0]}.csv"
        return pd.read_csv(alternate_file)

# Check for GPU availability
def check_gpu_availability():
    try:
        # Quick test to see if GPU is available for XGBoost
        # Updated to use newer XGBoost API to avoid warnings
        X = np.random.random((10, 5))
        y = np.random.randint(0, 2, 10)
        test_model = XGBClassifier(tree_method='hist', device='cuda:0')
        test_model.fit(X, y)
        return True
    except Exception:
        return False

# Create usable test data from NBA_test.csv and NBA_test_labels.csv.
def create_test_data():
    # Load the test data and labels
    test_data = pd.read_csv("data/NBA_test.csv")
    test_labels = pd.read_csv("data/NBA_test_labels.csv")

    # Create a copy to avoid modifying the original
    processed_data = test_data.copy()

    # Make sure we have the expected number of labels
    if len(test_labels) != len(test_data):
        print(f"Warning: Number of labels ({len(test_labels)}) does not match number of test rows ({len(test_data)})")

    # Process each row in the test data
    labels_used = 0  # Track labels used

    for idx, row in test_data.iterrows():
        if labels_used >= len(test_labels):
            print(f"Warning: Ran out of labels after {labels_used} rows. Some question marks may remain.")
            break

        # Case 1: Question mark is already in home_4
        if row['home_4'] == '?':
            # Simply replace it with the label
            processed_data.at[idx, 'home_4'] = test_labels.iloc[labels_used, 0]
            labels_used += 1
            continue

        # Case 2: Question mark is in one of home_0 through home_3
        question_mark_found = False
        for col_idx in range(4):  # 0 to 3
            col_name = f'home_{col_idx}'

            if row[col_name] == '?':
                # Swap the values: move home_4 to the column with ? and put ? in home_4
                processed_data.at[idx, col_name] = processed_data.at[idx, 'home_4']
                processed_data.at[idx, 'home_4'] = '?'

                # Replace the ? in home_4 with the label
                processed_data.at[idx, 'home_4'] = test_labels.iloc[labels_used, 0]
                labels_used += 1

                question_mark_found = True
                break

        # Case 3: No question mark in this row - nothing to do
        if not question_mark_found:
            continue

    # Save the processed data to a new CSV file
    processed_data_path = "data/NBA_test_data.csv"
    processed_data.to_csv(processed_data_path, index=False)

    # Return the processed data and the path to the file
    return processed_data, processed_data_path


# Function to prepare features for the model
def prepare_features(df_train, df_test=None, is_test_mode=False):
    """
    Prepare features for the model, including feature engineering and encoding.

    Args:
        df_train: Training DataFrame
        df_test: Testing DataFrame (optional)
        is_test_mode: Whether we're in test mode using NBA_test_data.csv

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_columns)
    """
    # Select only the relevant columns for modeling
    allowed_columns = ["home_team", "away_team", "home_0", "home_1", "home_2", "home_3", "home_4",
                       "away_0", "away_1", "away_2", "away_3", "away_4"]

    # Add outcome column for training data if it exists
    if "outcome" in df_train.columns:
        allowed_columns.append("outcome")

    df_train = df_train[allowed_columns].copy()

    if df_test is not None:
        df_test = df_test[[col for col in allowed_columns if col in df_test.columns]].copy()

    with timer("Feature engineering"):
        print("Feature engineering...")
        # Compute individual player win rates
        player_win_rates = {}
        if "outcome" in df_train.columns:
            player_win_rates_df = df_train.melt(id_vars=["outcome"],
                                                value_vars=["home_0", "home_1", "home_2", "home_3", "home_4",
                                                            "away_0", "away_1", "away_2", "away_3", "away_4"])
            player_win_rates = player_win_rates_df.groupby("value")["outcome"].mean().to_dict()

        # Map player win rates to both training and testing datasets
        for col in ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"]:
            if player_win_rates:
                df_train[col + "_win_rate"] = df_train[col].map(player_win_rates)

                # Fill NaN values with mean
                mean_win_rate = df_train[col + "_win_rate"].mean()
                df_train[col + "_win_rate"] = df_train[col + "_win_rate"].fillna(mean_win_rate)

                if df_test is not None:
                    df_test[col + "_win_rate"] = df_test[col].map(player_win_rates)
                    df_test[col + "_win_rate"] = df_test[col + "_win_rate"].fillna(mean_win_rate)
            else:
                # If no outcome column, use a default value of 0.5
                df_train[col + "_win_rate"] = 0.5
                if df_test is not None:
                    df_test[col + "_win_rate"] = 0.5

        # Compute synergy score using win ratio (if we have outcome data)
        def compute_synergy(df, players):
            synergy = {}
            if "outcome" in df.columns:
                for _, row in df.iterrows():
                    for i in range(len(players)):
                        for j in range(i + 1, len(players)):
                            pair = tuple(sorted([row[players[i]], row[players[j]]]))
                            if pair not in synergy:
                                synergy[pair] = {"wins": 0, "games": 0}
                            synergy[pair]["games"] += 1
                            if row["outcome"] == 1:
                                synergy[pair]["wins"] += 1
                synergy_scores = {k: v["wins"] / v["games"] if v["games"] > 0 else 0 for k, v in synergy.items()}
            else:
                # If no outcome data, use a default value of 0.5 for synergy
                synergy_scores = {}
            return synergy_scores

        # Compute synergy scores for the training dataset
        synergy_scores = compute_synergy(df_train,
                                         ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1",
                                          "away_2", "away_3", "away_4"])

        # Calculates the average synergy score for all unique player pairs
        def map_synergy_score(row, players):
            scores = []
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    pair = tuple(sorted([row[players[i]], row[players[j]]]))
                    scores.append(synergy_scores.get(pair, 0.5))  # Default to 0.5 if not found
            return np.mean(scores) if scores else 0.5

        # Adds synergy scores to train/test data by averaging player pair synergy.
        df_train["synergy_score"] = df_train.apply(lambda row: map_synergy_score(row, df_train.columns[2:12]), axis=1)
        if df_test is not None:
            df_test["synergy_score"] = df_test.apply(lambda row: map_synergy_score(row, df_test.columns[2:12]), axis=1)

        # Add team win rates (if we have outcome data)
        team_win_rates = {}
        if "outcome" in df_train.columns:
            team_win_rates = df_train.groupby("home_team")["outcome"].mean().to_dict()

        if team_win_rates:
            df_train["home_team_win_rate"] = df_train["home_team"].map(team_win_rates)

            away_team_win_rates = {k: 1 - v for k, v in team_win_rates.items()}
            df_train["away_team_win_rate"] = df_train["away_team"].map(away_team_win_rates)

            # Fill NaN values
            mean_team_win = df_train["home_team_win_rate"].mean()
            df_train["home_team_win_rate"] = df_train["home_team_win_rate"].fillna(mean_team_win)
            df_train["away_team_win_rate"] = df_train["away_team_win_rate"].fillna(1 - mean_team_win)

            if df_test is not None:
                df_test["home_team_win_rate"] = df_test["home_team"].map(team_win_rates)
                df_test["away_team_win_rate"] = df_test["away_team"].map(away_team_win_rates)
                df_test["home_team_win_rate"] = df_test["home_team_win_rate"].fillna(mean_team_win)
                df_test["away_team_win_rate"] = df_test["away_team_win_rate"].fillna(1 - mean_team_win)
        else:
            # If no outcome data, use a default value of 0.5
            df_train["home_team_win_rate"] = 0.5
            df_train["away_team_win_rate"] = 0.5
            if df_test is not None:
                df_test["home_team_win_rate"] = 0.5
                df_test["away_team_win_rate"] = 0.5

        # Player frequency features (how often a player appears)
        player_counts = {}
        for col in ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"]:
            for player in df_train[col].unique():
                if player not in player_counts:
                    player_counts[player] = 0
                player_counts[player] += df_train[col].value_counts().get(player, 0)

        # Normalize player frequency
        max_count = max(player_counts.values()) if player_counts else 1
        player_freq = {k: v / max_count for k, v in player_counts.items()}

        # Add player frequency to dataset
        for col in ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"]:
            df_train[col + "_freq"] = df_train[col].map(player_freq)

            # Fill NaN values with default or mean
            if len(df_train[col + "_freq"].dropna()) > 0:
                mean_freq = df_train[col + "_freq"].mean()
            else:
                mean_freq = 0.01  # Default small value if no data

            df_train[col + "_freq"] = df_train[col + "_freq"].fillna(mean_freq)

            if df_test is not None:
                df_test[col + "_freq"] = df_test[col].map(player_freq)
                df_test[col + "_freq"] = df_test[col + "_freq"].fillna(mean_freq)

        # Create additional features
        # Win rate differences for each position
        for i in range(5):  # For each position
            df_train[f'pos{i}_win_diff'] = df_train[f'home_{i}_win_rate'] - df_train[f'away_{i}_win_rate']

            df_train[f'pos{i}_freq_diff'] = df_train[f'home_{i}_freq'] - df_train[f'away_{i}_freq']

            if df_test is not None:
                df_test[f'pos{i}_win_diff'] = df_test[f'home_{i}_win_rate'] - df_test[f'away_{i}_win_rate']
                df_test[f'pos{i}_freq_diff'] = df_test[f'home_{i}_freq'] - df_test[f'away_{i}_freq']

        # Team composition statistics
        for team_type in ['home', 'away']:
            # Average player win rate for team
            df_train[f'{team_type}_avg_win'] = df_train[[f'{team_type}_{i}_win_rate' for i in range(5)]].mean(axis=1)

            # Standard deviation of player win rates (measure of team balance)
            df_train[f'{team_type}_std_win'] = df_train[[f'{team_type}_{i}_win_rate' for i in range(5)]].std(axis=1)

            # Min and max player win rates (weakest and strongest player)
            df_train[f'{team_type}_min_win'] = df_train[[f'{team_type}_{i}_win_rate' for i in range(5)]].min(axis=1)
            df_train[f'{team_type}_max_win'] = df_train[[f'{team_type}_{i}_win_rate' for i in range(5)]].max(axis=1)

            if df_test is not None:
                df_test[f'{team_type}_avg_win'] = df_test[[f'{team_type}_{i}_win_rate' for i in range(5)]].mean(axis=1)
                df_test[f'{team_type}_std_win'] = df_test[[f'{team_type}_{i}_win_rate' for i in range(5)]].std(axis=1)
                df_test[f'{team_type}_min_win'] = df_test[[f'{team_type}_{i}_win_rate' for i in range(5)]].min(axis=1)
                df_test[f'{team_type}_max_win'] = df_test[[f'{team_type}_{i}_win_rate' for i in range(5)]].max(axis=1)

        # Team vs team comparative features
        df_train['team_win_diff'] = df_train['home_team_win_rate'] - df_train['away_team_win_rate']
        df_train['avg_win_diff'] = df_train['home_avg_win'] - df_train['away_avg_win']

        if df_test is not None:
            df_test['team_win_diff'] = df_test['home_team_win_rate'] - df_test['away_team_win_rate']
            df_test['avg_win_diff'] = df_test['home_avg_win'] - df_test['away_avg_win']

        # Encode categorical features
        encoder = LabelEncoder()
        categorical_columns = ["home_team", "away_team", "home_0", "home_1", "home_2", "home_3",
                               "away_0", "away_1", "away_2", "away_3", "away_4"]

        # Apply encoding and handle unknown values in test set
        for col in categorical_columns + ["home_4"]:
            # Fit the encoder on the training data
            df_train[col + "_encoded"] = encoder.fit_transform(df_train[col])

            if df_test is not None:
                # Handle unknown values in test set
                most_common_value = df_train[col].mode()[0]

                # Vectorized transformation for test set
                test_values = np.array(df_test[col])
                encoded_values = np.zeros(len(test_values), dtype=int)

                # For values in encoder classes, use the encoder
                mask_in_classes = np.isin(test_values, encoder.classes_)
                values_in_classes = test_values[mask_in_classes]
                if len(values_in_classes) > 0:
                    encoded_values[mask_in_classes] = encoder.transform(values_in_classes)

                # For unknown values, use the most common class
                if sum(~mask_in_classes) > 0:
                    encoded_values[~mask_in_classes] = encoder.transform([most_common_value])[0]

                df_test[col + "_encoded"] = encoded_values

        # Scale numerical features
        numerical_cols = [col for col in df_train.columns if
                          col.endswith('_win_rate') or
                          col.endswith('_freq') or
                          col.endswith('_diff') or
                          col.endswith('_std_win') or
                          col.endswith('_min_win') or
                          col.endswith('_max_win') or
                          col == 'synergy_score']

        scaler = StandardScaler()
        df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
        if df_test is not None:
            df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    # Create feature set for model training
    feature_columns = [col for col in df_train.columns if
                       col.endswith('_encoded') or
                       col.endswith('_win_rate') or
                       col.endswith('_freq') or
                       col.endswith('_diff') or
                       col.endswith('_std_win') or
                       col.endswith('_min_win') or
                       col.endswith('_max_win') or
                       col == 'synergy_score']

    # Create parameters to train model
    X_train = df_train.drop(columns=["home_4"] + (["outcome"] if "outcome" in df_train.columns else []))
    y_train = df_train["home_4_encoded"]

    X_test = None
    y_test = None

    if df_test is not None:
        test_columns_to_drop = ["home_4"]
        if "outcome" in df_test.columns:
            test_columns_to_drop.append("outcome")

        X_test = df_test.drop(columns=[col for col in test_columns_to_drop if col in df_test.columns])
        if "home_4_encoded" in df_test.columns:
            y_test = df_test["home_4_encoded"]

    # Select only the required features
    X_train = X_train[feature_columns]
    if X_test is not None:
        X_test = X_test[[col for col in feature_columns if col in X_test.columns]]

    return X_train, y_train, X_test, y_test, feature_columns


# Function to train model and make predictions
def train_and_predict(X_train, y_train, X_test, use_gpu=False):
    """
    Train a model and make predictions.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        use_gpu: Whether to use GPU for training

    Returns:
        Tuple of (model, top_1_predictions, top_3_predictions)
    """
    # Configure XGBoost based on GPU availability
    if use_gpu:
        xgboost_params = {
            'tree_method': 'hist',
            'device': 'cuda:0',
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
    else:
        xgboost_params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }

    # Initialize and train model
    with timer("Model training"):
        print("Training model...")
        model = XGBClassifier(**xgboost_params)
        model.fit(X_train, y_train)

    # Generate predictions
    with timer("Prediction"):
        print("Generating predictions...")
        # Get predicted probabilities for test data
        y_prob = model.predict_proba(X_test)

        # Extract indices of top 3 predicted classes per row
        top_3_indices = np.argsort(y_prob, axis=1)[:, -3:]

        # Get top 1 predictions (highest probability per row)
        y_pred_top1 = model.predict(X_test)

        # Convert indices to actual player class labels
        top_3_preds = np.array(model.classes_)[top_3_indices]

    return model, y_pred_top1, top_3_preds


# Function to train model and make predictions
def train_and_predict(X_train, y_train, X_test, use_gpu=False):
    """
    Train a model and make predictions.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        use_gpu: Whether to use GPU for training

    Returns:
        Tuple of (model, top_1_predictions, top_3_predictions)
    """
    # Configure XGBoost based on GPU availability
    if use_gpu:
        xgboost_params = {
            'tree_method': 'hist',
            'device': 'cuda:0',
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
    else:
        xgboost_params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }

    # Initialize and train model
    with timer("Model training"):
        print("Model training...")
        model = XGBClassifier(**xgboost_params)
        model.fit(X_train, y_train)

    # Generate predictions
    with timer("Generating predictions"):
        print("Generating predictions...")
        # Get predicted probabilities for test data
        y_prob = model.predict_proba(X_test)

        # Extract indices of top 3 predicted classes per row
        top_3_indices = np.argsort(y_prob, axis=1)[:, -3:]

        # Get top 1 predictions (highest probability per row)
        y_pred_top1 = model.predict(X_test)

        # Convert indices to actual player class labels
        top_3_preds = np.array(model.classes_)[top_3_indices]

    return model, y_pred_top1, top_3_preds


# Function to create visualization charts
def create_visualizations(results_df):
    """
    Create visualization charts for the results.

    Args:
        results_df: DataFrame containing the results
    """
    # Create a directory for saving the visualization charts
    os.makedirs("visualization_charts", exist_ok=True)

    # Helper function to style plots
    def style_plot(ax, title, ylabel, legend_title):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Year → Testing Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 1.0)  # All metrics are between 0 and 1
        ax.grid(True, linestyle='--', alpha=0.7)

        # If no legend exists yet, create one (some plots specify their own legends)
        if not ax.get_legend():
            ax.legend(title=legend_title)

        # Format x-axis labels as "2007→2008"
        x_labels = [f"{row['Train Year']}→{row['Test Year']}" for _, row in results_df.iterrows()]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)

        # Add horizontal lines for mean values
        handles, labels = ax.get_legend_handles_labels()

        # Fix the metric mapping to ensure proper correspondence
        metric_mapping = {
            'Top 1 Accuracy': 'Top 1 Accuracy',
            'Top 3 Accuracy': 'Top 3 Accuracy',
            'Top 1 Precision': 'Top 1 Precision',
            'Top 3 Precision': 'Top 3 Precision',
            'Top 1 Recall': 'Top 1 Recall',
            'Top 3 Recall': 'Top 3 Recall',
            'Top 1 F1-score': 'Top 1 F1-score',
            'Top 1 Accuracy/Recall (identical)': 'Top 1 Recall',
            'Top 3 Accuracy/Recall (identical)': 'Top 3 Recall',
            'Top 3 F1-score': 'Top 3 F1-score'
        }

        for i, (handle, label) in enumerate(zip(handles, labels)):
            # Find the matching column name
            col_name = None
            for pattern, column in metric_mapping.items():
                if pattern in label:
                    col_name = column
                    break

            # If no match found in our mapping, try to find it in the DataFrame columns
            if col_name is None:
                for col in results_df.columns:
                    if label in col:
                        col_name = col
                        break

            # If we found a matching column name, add the mean line
            if col_name and col_name in results_df.columns:
                mean_val = results_df[col_name].mean()
                color = handle.get_color()
                ax.axhline(y=mean_val, color=color, linestyle=':', alpha=0.7)
                ax.text(len(results_df) - 1, mean_val + 0.02, f'Mean: {mean_val:.3f}',
                        color=color, fontsize=9, ha='right')

        return ax

    # 1. Create Accuracy Chart
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # Explicitly include both lines with distinct colors that match the other charts
    line1 = ax.plot(results_df['Top 1 Accuracy'], 'o-', color='#1f77b4', label='Top 1 Accuracy')[0]
    line2 = ax.plot(results_df['Top 3 Accuracy'], 's-', color='#ff7f0e', label='Top 3 Accuracy')[0]

    # Ensure the legend displays both lines
    ax.legend(handles=[line1, line2], title='Metrics')

    style_plot(ax, 'Prediction Accuracy Across Years', 'Accuracy Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/accuracy_chart.png', dpi=300)
    plt.close()

    # 2. Create Precision Chart
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # Explicitly include both Top 1 and Top 3 Precision with distinct markers and colors
    line1 = ax.plot(results_df['Top 1 Precision'], 'o-', color='#1f77b4', label='Top 1 Precision')[0]
    line2 = ax.plot(results_df['Top 3 Precision'], 's-', color='#ff7f0e', label='Top 3 Precision')[0]

    # Ensure the legend displays both lines
    ax.legend(handles=[line1, line2], title='Metrics')

    style_plot(ax, 'Prediction Precision Across Years', 'Precision Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/precision_chart.png', dpi=300)
    plt.close()

    # 3. Create Recall Chart
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # Explicitly include both Top 1 and Top 3 Recall with distinct markers and colors
    line1 = ax.plot(results_df['Top 1 Recall'], 'o-', color='#1f77b4', label='Top 1 Recall')[0]
    line2 = ax.plot(results_df['Top 3 Recall'], 's-', color='#ff7f0e', label='Top 3 Recall')[0]

    # Ensure the legend displays both lines
    ax.legend(handles=[line1, line2], title='Metrics')

    style_plot(ax, 'Prediction Recall Across Years', 'Recall Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/recall_chart.png', dpi=300)
    plt.close()

    # 4. Create F1-Score Chart
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # Explicitly include both lines with consistent colors
    line1 = ax.plot(results_df['Top 1 F1-score'], 'o-', color='#1f77b4', label='Top 1 F1-score')[0]
    line2 = ax.plot(results_df['Top 3 F1-score'], 's-', color='#ff7f0e', label='Top 3 F1-score')[0]

    # Ensure the legend displays both lines
    ax.legend(handles=[line1, line2], title='Metrics')

    style_plot(ax, 'Prediction F1-Score Across Years', 'F1-Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/f1_score_chart.png', dpi=300)
    plt.close()

    # 5. Create a combined metrics chart for Top 1 predictions
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    # Define consistent colors for each metric type across all charts
    accuracy_color = '#1f77b4'  # blue
    precision_color = '#ff7f0e'  # orange
    recall_color = '#2ca02c'  # green
    f1_color = '#d62728'  # red

    # Plot with explicit colors - but combine accuracy and recall in the legend
    # Use recall data but label it as both accuracy and recall since they're identical
    line1 = ax.plot(results_df['Top 1 Recall'], 'o-', color=accuracy_color,
                    label='Top 1 Accuracy/Recall (identical)')[0]
    line2 = ax.plot(results_df['Top 1 Precision'], 's-', color=precision_color,
                    label='Top 1 Precision')[0]
    line3 = ax.plot(results_df['Top 1 F1-score'], 'd-', color=f1_color,
                    label='Top 1 F1-score')[0]

    # Ensure the legend displays all lines - we've removed the duplicate recall line
    ax.legend(handles=[line1, line2, line3], title='Metrics')

    style_plot(ax, 'Top 1 Prediction Metrics Across Years', 'Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/top1_combined_metrics.png', dpi=300)
    plt.close()

    # 6. Create a combined metrics chart for Top 3 predictions
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    # Use the same colors for consistency - but combine accuracy and recall in the legend
    line1 = ax.plot(results_df['Top 3 Recall'], 'o-', color=accuracy_color,
                    label='Top 3 Accuracy/Recall (identical)')[0]
    line2 = ax.plot(results_df['Top 3 Precision'], 's-', color=precision_color,
                    label='Top 3 Precision')[0]
    line3 = ax.plot(results_df['Top 3 F1-score'], 'd-', color=f1_color,
                    label='Top 3 F1-score')[0]

    # Ensure the legend displays all lines - we've removed the duplicate recall line
    ax.legend(handles=[line1, line2, line3], title='Metrics')

    style_plot(ax, 'Top 3 Prediction Metrics Across Years', 'Score', 'Metrics')
    plt.tight_layout()
    plt.savefig('visualization_charts/top3_combined_metrics.png', dpi=300)
    plt.close()

    print("\nVisualization charts have been created in the 'visualization_charts' directory:")
    print("- accuracy_chart.png: Top 1 and Top 3 accuracy comparison")
    print("- precision_chart.png: Top 1 and Top 3 precision comparison")
    print("- recall_chart.png: Top 1 and Top 3 recall comparison")
    print("- f1_score_chart.png: Top 1 and Top 3 F1-Score comparison")
    print("- top1_combined_metrics.png: All metrics for Top 1 predictions")
    print("- top3_combined_metrics.png: All metrics for Top 3 predictions")


# Function to run the original year pairs mode
def run_year_pairs_mode():
    print("\nRunning model using training/testing year pairs...\n")

    # Store prediction results
    results = []

    # Check GPU availability
    use_gpu = check_gpu_availability()

    # Loop for each training/testing years pair
    for i in range(len(train_test_years) - 1):
        # Assign training/testing year values
        train_year = train_test_years[i]
        test_year = train_test_years[i + 1]

        print(f"\nProcessing Training Year: {train_year} -> Testing Year: {test_year}")

        # Identify data set file names for training/testing years
        train_file = f"data/matchups-{train_year}.csv"
        test_file = f"data/matchups-{test_year}.csv"

        # Load CSV files into panda dataframes
        with timer("Data loading"):
            print("Loading data...")
            df_train = load_data(train_file)
            df_test = load_data(test_file)

        # Prepare features
        X_train, y_train, X_test, y_test, feature_columns = prepare_features(df_train, df_test)

        # Train model and make predictions
        model, y_pred_top1, top_3_preds = train_and_predict(X_train, y_train, X_test, use_gpu)

        # Evaluate predictions
        top_1_accuracy = accuracy_score(y_test, y_pred_top1)
        top_1_precision = precision_score(y_test, y_pred_top1, average='weighted', zero_division=0)
        top_1_recall = recall_score(y_test, y_pred_top1, average='weighted', zero_division=0)
        top_1_f1 = f1_score(y_test, y_pred_top1, average='weighted', zero_division=0)

        # Calculate top 3 metrics
        # Calculate if the true label is in any of the top 3 predictions (this is the recall@3)
        top_3_correct = [y_test.iloc[i] in top_3_preds[i] for i in range(len(y_test))]
        top_3_accuracy = np.mean(top_3_correct)

        # For top-3 metrics in single-label classification:
        # Recall@3: Same as top_3_accuracy (did we include the true label in our 3 guesses?)
        top_3_recall = top_3_accuracy

        # Precision@3: Since we're making 3 guesses for a single correct label,
        # precision is diluted by a factor of 3 (or by the number of predictions we actually made)
        top_3_precision = top_3_accuracy / 3

        # F1@3: Harmonic mean of precision@3 and recall@3
        if top_3_precision + top_3_recall > 0:  # Avoid division by zero
            top_3_f1 = 2 * (top_3_precision * top_3_recall) / (top_3_precision + top_3_recall)
        else:
            top_3_f1 = 0.0

        # Store evaluation metrics for current train-test year pair
        results.append((train_year, test_year, top_1_accuracy, top_1_precision, top_1_recall, top_1_f1, top_3_accuracy,
                        top_3_precision, top_3_recall, top_3_f1))

        # Print results
        print(f"\nResults for Training Year: {train_year} -> Testing Year: {test_year}")
        print(
            f"Top 1 Accuracy: {top_1_accuracy:.4f}, Precision: {top_1_precision:.4f}, Recall: {top_1_recall:.4f}, F1-score: {top_1_f1:.4f}")
        print(
            f"Top 3 Accuracy: {top_3_accuracy:.4f}, Precision: {top_3_precision:.4f}, Recall: {top_3_recall:.4f}, F1-score: {top_3_f1:.4f}")
        print("=" * 50)

    # Save results to CSV
    results_df = pd.DataFrame(results,
                              columns=["Train Year", "Test Year", "Top 1 Accuracy", "Top 1 Precision", "Top 1 Recall",
                                       "Top 1 F1-score", "Top 3 Accuracy", "Top 3 Precision", "Top 3 Recall",
                                       "Top 3 F1-score"])
    results_df.to_csv("data/results.csv", index=False)

    # Print summary statistics
    print("\nSummary of XGBoost performance across all years:")
    print(f"Mean Top 1 Accuracy: {results_df['Top 1 Accuracy'].mean():.4f}")
    print(f"Mean Top 1 F1-score: {results_df['Top 1 F1-score'].mean():.4f}")
    print(f"Mean Top 3 Accuracy: {results_df['Top 3 Accuracy'].mean():.4f}")
    print(f"Mean Top 3 F1-score: {results_df['Top 3 F1-score'].mean():.4f}")

    print("Results saved to results.csv")

    # Create visualizations
    create_visualizations(results_df)


def run_test_data_mode():
    """
    Run model using test data (NBA_test.csv).
    MODIFIED: Now ensures consistent encoding between modes
    """
    print("\nRunning model using test data (NBA_test.csv)...\n")

    # Create test data
    processed_data, processed_data_path = create_test_data()

    # Check GPU availability
    use_gpu = check_gpu_availability()

    # Load all training data from 2007-2015
    with timer("Loading all training data from 2007-2015"):
        print("Loading training data...")
        all_training_data = []

        # Dictionary to track players by season
        players_by_season = {}

        for year in train_test_years:
            file_path = f"data/matchups-{year}.csv"
            try:
                df = load_data(file_path)
                all_training_data.append(df)

                # Extract unique players for this season
                season_players = set()
                for col in ["home_0", "home_1", "home_2", "home_3", "home_4",
                            "away_0", "away_1", "away_2", "away_3", "away_4"]:
                    season_players.update(df[col].unique())

                players_by_season[year] = season_players

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        combined_training_data = pd.concat(all_training_data, ignore_index=True)

    # Load the processed test data
    test_data = pd.read_csv(processed_data_path)

    # Extract actual values and seasons from test data
    actual_values = []
    test_seasons = []
    for idx, row in test_data.iterrows():
        actual_values.append(row['home_4'])
        test_seasons.append(row['season'])

    # Convert to numpy arrays for easier processing
    actual_values = np.array(actual_values)
    test_seasons = np.array(test_seasons)

    # CRITICAL FIX: Use the exact same prepare_features function as in year pairs mode
    # This ensures unknown players are handled consistently between modes
    X_train, y_train, X_test, _, feature_columns = prepare_features(combined_training_data, test_data,
                                                                    is_test_mode=True)

    # Train model and make predictions
    model, y_pred_top1, top_3_preds = train_and_predict(X_train, y_train, X_test, use_gpu)

    # Initialize timing and progress reporting
    print("\nStarting post-processing of predictions...")

    # Convert encoded predictions back to player names
    encoder = LabelEncoder()
    encoder.fit(combined_training_data['home_4'])

    # Pre-compute encoder mappings once to avoid repeated conversions
    print("Building encoder mappings...")
    start_time = time.time()
    encoder_classes = np.array(encoder.classes_)
    encoded_to_player = {i: player for i, player in enumerate(encoder_classes)}
    player_to_encoded = {player: i for i, player in enumerate(encoder_classes)}
    print(f"Encoder mappings built in {time.time() - start_time:.2f} seconds")

    # Get prediction probabilities
    print("Getting prediction probabilities...")
    start_time = time.time()
    y_probs = model.predict_proba(X_test)
    print(f"Probabilities retrieved in {time.time() - start_time:.2f} seconds")

    # Initialize arrays
    predictions = []
    filtered_top3_list = []

    # Process predictions in batches to avoid memory issues
    batch_size = 100
    total_rows = len(y_pred_top1)
    num_batches = (total_rows + batch_size - 1) // batch_size

    print(f"Processing {total_rows} predictions in {num_batches} batches of {batch_size}...")

    total_start_time = time.time()
    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)

        print(f"Processing batch {batch_idx + 1}/{num_batches} (records {start_idx + 1}-{end_idx})...")

        for i in range(start_idx, end_idx):
            # Get current season
            season = test_seasons[i]
            season_player_set = players_by_season.get(season, set())

            # Process top-1 prediction
            pred = y_pred_top1[i]
            orig_player = encoded_to_player[pred]

            if orig_player in season_player_set:
                # Keep original prediction if it's from the correct season
                predictions.append(pred)
            else:
                # Find valid alternatives
                valid_indices = []
                for idx, player in enumerate(encoder_classes):
                    if player in season_player_set:
                        valid_indices.append(idx)

                if valid_indices:
                    # Get most likely valid player
                    probs = y_probs[i]
                    best_idx = valid_indices[0]
                    best_prob = probs[best_idx]

                    for idx in valid_indices[1:]:
                        if probs[idx] > best_prob:
                            best_idx = idx
                            best_prob = probs[idx]

                    predictions.append(best_idx)
                else:
                    # Keep original if no valid alternatives
                    predictions.append(pred)

            # Process top-3 predictions
            top3_pred = top_3_preds[i]
            filtered_top3 = []

            # Keep original predictions if valid
            for idx in top3_pred:
                player = encoded_to_player[idx]
                if player in season_player_set:
                    filtered_top3.append(idx)

            # Add more valid players if needed
            if len(filtered_top3) < 3:
                # Get additional players from this season
                additional_indices = []
                probs = y_probs[i]

                for idx, player in enumerate(encoder_classes):
                    if player in season_player_set and idx not in filtered_top3:
                        additional_indices.append((idx, probs[idx]))

                # Sort by probability and add top ones
                if additional_indices:
                    additional_indices.sort(key=lambda x: x[1], reverse=True)
                    needed = 3 - len(filtered_top3)
                    filtered_top3.extend([idx for idx, _ in additional_indices[:needed]])

            # Ensure exactly 3 predictions when possible
            filtered_top3 = filtered_top3[:3]
            while len(filtered_top3) < 3 and len(filtered_top3) < len(top3_pred):
                filtered_top3.append(top3_pred[len(filtered_top3)])

            filtered_top3_list.append(filtered_top3)

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        print(f"Batch {batch_idx + 1} completed in {batch_duration:.2f} seconds")

        # Estimate remaining time
        avg_time_per_batch = (batch_end_time - total_start_time) / (batch_idx + 1)
        remaining_batches = num_batches - (batch_idx + 1)
        estimated_time = avg_time_per_batch * remaining_batches

        print(f"Estimated time remaining: {estimated_time:.2f} seconds")

    # Convert to numpy arrays
    predictions = np.array(predictions)
    filtered_top3_list = [np.array(x) for x in filtered_top3_list]

    print(f"Post-processing completed in {time.time() - total_start_time:.2f} seconds")

    # Convert predictions to player names
    print("Converting predictions to player names...")
    start_time = time.time()
    top_1_player_names = []

    for pred in predictions:
        top_1_player_names.append(encoded_to_player.get(pred, "Unknown"))

    # Convert top-3 predictions
    top_3_player_names = []
    for preds in filtered_top3_list:
        players = []
        for pred in preds:
            players.append(encoded_to_player.get(pred, "Unknown"))
        top_3_player_names.append(players)

    print(f"Conversion completed in {time.time() - start_time:.2f} seconds")

    # Create results DataFrame
    print("Creating results DataFrame...")
    start_time = time.time()
    results = pd.DataFrame({
        'Season': test_seasons,
        'Actual': actual_values,
        'Predicted': top_1_player_names
    })

    # Add top 3 predictions as separate columns
    for i in range(3):
        # Handle cases where some top 3 arrays might have fewer than 3 elements
        results[f'Top_{i + 1}'] = [
            players[i] if i < len(players) else "Unknown"
            for players in top_3_player_names
        ]
    print(f"DataFrame created in {time.time() - start_time:.2f} seconds")

    # Calculate metrics
    print("Calculating evaluation metrics...")
    start_time = time.time()

    # Calculate if top 1 prediction is correct
    results['Top_1_Correct'] = results['Actual'] == results['Predicted']

    # Calculate if actual value is in top 3 predictions
    results['In_Top_3'] = False
    for idx, row in results.iterrows():
        actual = row['Actual']
        top3 = [row[f'Top_{i + 1}'] for i in range(3)]
        if actual in top3:
            results.at[idx, 'In_Top_3'] = True

    # Calculate accuracy metrics
    top_1_accuracy = results['Top_1_Correct'].mean()
    top_3_accuracy = results['In_Top_3'].mean()

    # Convert actual and predicted values to encoded values for scikit-learn metrics
    actual_encoded = np.array([player_to_encoded.get(player, -1) for player in actual_values])
    predicted_encoded = np.array([player_to_encoded.get(player, -1) for player in top_1_player_names])

    # Handle any potential mapping issues
    valid_indices = (actual_encoded != -1) & (predicted_encoded != -1)
    if not all(valid_indices):
        print(
            f"Warning: {np.sum(~valid_indices)} records had mapping issues and will be excluded from precision/recall metrics.")

    # Calculate precision, recall, and F1-score for Top-1 predictions
    if np.sum(valid_indices) > 0:
        top_1_precision = precision_score(
            actual_encoded[valid_indices],
            predicted_encoded[valid_indices],
            average='weighted',
            zero_division=0
        )
        top_1_recall = recall_score(
            actual_encoded[valid_indices],
            predicted_encoded[valid_indices],
            average='weighted',
            zero_division=0
        )
        top_1_f1 = f1_score(
            actual_encoded[valid_indices],
            predicted_encoded[valid_indices],
            average='weighted',
            zero_division=0
        )
    else:
        top_1_precision = top_1_recall = top_1_f1 = 0.0

    # Calculate Top-3 metrics
    # For top-3 metrics, recall@3 is the same as top_3_accuracy (did we include the true label in our 3 guesses?)
    top_3_recall = top_3_accuracy

    # Precision@3: Since we're making 3 guesses for a single correct label,
    # precision is diluted by a factor of 3 (or by the number of predictions we actually made)
    top_3_precision = top_3_accuracy / 3

    # F1@3: Harmonic mean of precision@3 and recall@3
    if top_3_precision + top_3_recall > 0:  # Avoid division by zero
        top_3_f1 = 2 * (top_3_precision * top_3_recall) / (top_3_precision + top_3_recall)
    else:
        top_3_f1 = 0.0
    print(f"Metrics calculated in {time.time() - start_time:.2f} seconds")

    # Save all metrics to the results
    metrics = {
        'Top_1_Accuracy': top_1_accuracy,
        'Top_1_Precision': top_1_precision,
        'Top_1_Recall': top_1_recall,
        'Top_1_F1': top_1_f1,
        'Top_3_Accuracy': top_3_accuracy,
        'Top_3_Precision': top_3_precision,
        'Top_3_Recall': top_3_recall,
        'Top_3_F1': top_3_f1
    }

    # Save results and metrics
    print("Saving results to files...")
    start_time = time.time()
    results.to_csv("data/test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv("data/test_metrics.csv", index=False)
    print(f"Results saved in {time.time() - start_time:.2f} seconds")

    # Print results
    print("\nResults for NBA Test Data:")
    print(f"Top 1 Accuracy: {top_1_accuracy:.4f}")
    print(f"Top 1 Precision: {top_1_precision:.4f}")
    print(f"Top 1 Recall: {top_1_recall:.4f}")
    print(f"Top 1 F1-score: {top_1_f1:.4f}")
    print(f"Top 3 Accuracy: {top_3_accuracy:.4f}")
    print(f"Top 3 Precision: {top_3_precision:.4f}")
    print(f"Top 3 Recall: {top_3_recall:.4f}")
    print(f"Top 3 F1-score: {top_3_f1:.4f}")
    print("Detailed predictions saved to data/test_predictions.csv")
    print("Metrics summary saved to data/test_metrics.csv")


# Main function
def main():
    print("=" * 50)
    print("NBA Lineup Prediction Program")

    while True:
        print("=" * 50)
        print("Options:")
        print("1. Run model using test data (NBA_test.csv)")
        print("2. Run model using training/testing year pairs (original functionality)")
        print("3. Exit")
        print("=" * 50)
        choice = input("\nEnter your choice (1, 2, or 3): ")
        if choice == '1':
            run_test_data_mode()
        if choice == '2':
            run_year_pairs_mode()
        if choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print("\nProgram execution completed.")

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()