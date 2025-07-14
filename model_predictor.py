import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import logging
import os
from config import MIN_MODEL_PROBABILITY, MIN_MODEL_EXPECTED_VALUE, MIN_PROBABILITY_DIFFERENCE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPredictor:
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.features = [
            'player1_id_numeric', 'player2_id_numeric',
            'odds_player1', 'odds_player2'
        ]
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logging.info("Model and scaler loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading model or scaler: {e}")
                self.model = None
                self.scaler = None
        else:
            logging.warning("Model or scaler files not found. Model needs to be trained.")

    def preprocess_data(self, df):
        if df.empty:
            logging.warning("Empty DataFrame provided for preprocessing.")
            return pd.DataFrame()

        df_copy = df.copy()

        player_ids = pd.unique(df_copy[['player1_id', 'player2_id']].values.ravel('K'))
        player_id_map = {id: i for i, id in enumerate(player_ids)}

        df_copy['player1_id_numeric'] = df_copy['player1_id'].map(player_id_map).fillna(-1).astype(int)
        df_copy['player2_id_numeric'] = df_copy['player2_id'].map(player_id_map).fillna(-1).astype(int)
        
        df_copy['odds_player1'] = pd.to_numeric(df_copy['odds_player1'], errors='coerce')
        df_copy['odds_player2'] = pd.to_numeric(df_copy['odds_player2'], errors='coerce')

        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_copy.fillna(0, inplace=True)

        return df_copy

    def train_model(self, df_training_data, model_path, scaler_path):
        if df_training_data.empty:
            logging.error("Cannot train model: No training data provided.")
            return

        df_processed = self.preprocess_data(df_training_data.copy())
        
        df_processed = df_processed[df_processed['status'] == 'Finished'].copy()
        if df_processed.empty:
            logging.warning("No finished matches in training data after filtering. Cannot train model.")
            return

        # Ensure winner_id, player1_id, player2_id are present for target creation
        if 'winner_id' not in df_processed.columns or 'player1_id' not in df_processed.columns or 'player2_id' not in df_processed.columns:
            logging.error("Missing 'winner_id', 'player1_id', or 'player2_id' for target creation. Cannot train model.")
            return

        df_processed['target'] = df_processed.apply(
            lambda row: 1 if str(row['winner_id']) == str(row['player1_id']) else (0 if str(row['winner_id']) == str(row['player2_id']) else np.nan), axis=1
        )
        df_processed.dropna(subset=['target'], inplace=True)
        
        if df_processed.empty:
            logging.warning("No valid targets after dropping NaNs. Cannot train model.")
            return

        X = df_processed[self.features]
        y = df_processed['target']

        if X.empty or y.empty or len(X) < 2:
            logging.error("Insufficient data to train the model after preprocessing and target assignment.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model.fit(X_train_scaled, y_train)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Model and scaler saved to {model_path} and {scaler_path}.")
        
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        logging.info(f"Model training complete. Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

    def predict(self, df_data):
        if df_data.empty:
            logging.warning("Empty DataFrame provided for prediction.")
            return pd.DataFrame()

        df_processed = df_data.copy()
        
        if 'player1_id_numeric' not in df_processed.columns or 'player2_id_numeric' not in df_processed.columns:
            logging.warning("Player numeric IDs not found. Preprocessing again for prediction.")
            df_processed = self.preprocess_data(df_processed)

        missing_features = [f for f in self.features if f not in df_processed.columns]
        if missing_features:
            logging.error(f"Missing features for prediction: {missing_features}. Cannot predict.")
            return pd.DataFrame()

        # Initialize prediction columns with default values
        df_processed['model_probability_player1'] = np.nan
        df_processed['model_probability_player2'] = np.nan
        df_processed['implied_odds_player1'] = np.nan
        df_processed['implied_odds_player2'] = np.nan
        df_processed['model_expected_value_player1'] = np.nan
        df_processed['model_expected_value_player2'] = np.nan
        df_processed['prob_diff_player1'] = np.nan
        df_processed['prob_diff_player2'] = np.nan
        df_processed['recommended_bet_on'] = None
        df_processed['recommended_odds'] = np.nan
        df_processed['recommended_probability'] = np.nan
        df_processed['recommended_expected_value'] = np.nan
        df_processed['recommended_prob_diff'] = np.nan
        df_processed['prediction_status'] = 'Not Predicted (Model Not Loaded)'

        if self.model is None or self.scaler is None:
            logging.warning("Model or scaler not loaded. Cannot make full predictions.")
            return df_processed # Return with default NaN/None values

        try:
            X_predict = df_processed[self.features]
            X_predict_scaled = self.scaler.transform(X_predict)

            probabilities = self.model.predict_proba(X_predict_scaled)
            
            df_processed['model_probability_player1'] = probabilities[:, 1]
            df_processed['model_probability_player2'] = probabilities[:, 0]

            df_processed['implied_odds_player1'] = 1 / df_processed['odds_player1'].replace(0, np.nan)
            df_processed['implied_odds_player2'] = 1 / df_processed['odds_player2'].replace(0, np.nan)

            df_processed['model_expected_value_player1'] = (df_processed['model_probability_player1'] * df_processed['odds_player1']) - 1
            df_processed['model_expected_value_player2'] = (df_processed['model_probability_player2'] * df_processed['odds_player2']) - 1

            df_processed['prob_diff_player1'] = df_processed['model_probability_player1'] - (1 / df_processed['odds_player1'])
            df_processed['prob_diff_player2'] = df_processed['model_probability_player2'] - (1 / df_processed['odds_player2'])

            for index, row in df_processed.iterrows():
                if row['model_expected_value_player1'] > row['model_expected_value_player2']:
                    df_processed.loc[index, 'recommended_bet_on'] = row['player1_name']
                    df_processed.loc[index, 'recommended_odds'] = row['odds_player1']
                    df_processed.loc[index, 'recommended_probability'] = row['model_probability_player1']
                    df_processed.loc[index, 'recommended_expected_value'] = row['model_expected_value_player1']
                    df_processed.loc[index, 'recommended_prob_diff'] = row['prob_diff_player1']
                    df_processed.loc[index, 'prediction_status'] = 'Predicted'
                elif row['model_expected_value_player2'] > row['model_expected_value_player1']:
                    df_processed.loc[index, 'recommended_bet_on'] = row['player2_name']
                    df_processed.loc[index, 'recommended_odds'] = row['odds_player2']
                    df_processed.loc[index, 'recommended_probability'] = row['model_probability_player2']
                    df_processed.loc[index, 'recommended_expected_value'] = row['model_expected_value_player2']
                    df_processed.loc[index, 'recommended_prob_diff'] = row['prob_diff_player2']
                    df_processed.loc[index, 'prediction_status'] = 'Predicted'
                else:
                    df_processed.loc[index, 'prediction_status'] = 'No Strong Bet'
            
            return df_processed

        except Exception as e:
            logging.error(f"Error during prediction process: {e}")
            df_processed['prediction_status'] = 'Prediction Failed'
            return df_processed

    def filter_predictions(self, df_predictions):
        if df_predictions.empty:
            return pd.DataFrame()

        # Ensure required columns for filtering exist, fill with 0/NaN if not
        filter_check_cols = ['recommended_probability', 'recommended_expected_value', 'recommended_prob_diff']
        for col in filter_check_cols:
            if col not in df_predictions.columns:
                df_predictions[col] = np.nan # Use NaN for numerical columns

        df_filtered = df_predictions[
            (df_predictions["recommended_probability"] >= MIN_MODEL_PROBABILITY) &
            (df_predictions["recommended_expected_value"] >= MIN_MODEL_EXPECTED_VALUE) &
            (df_predictions["recommended_prob_diff"] >= MIN_PROBABILITY_DIFFERENCE)
        ].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()

        output_cols = [
            'event_date', 'event_time', 'tournament_name',
            'player1_name', 'player2_name', 'odds_player1', 'odds_player2',
            'recommended_bet_on', 'recommended_odds',
            'recommended_probability', 'recommended_expected_value', 'recommended_prob_diff',
            'match_id', 'player1_id', 'player2_id', 'status'
        ]
        
        # Ensure all output_cols exist before selecting
        for col in output_cols:
            if col not in df_filtered.columns:
                df_filtered[col] = None # Add missing column with None

        df_filtered = df_filtered[output_cols].copy()
        df_filtered.rename(columns={
            'event_date': 'Match Date',
            'event_time': 'Match Time',
            'tournament_name': 'Tournament',
            'player1_name': 'Player 1',
            'player2_name': 'Player 2',
            'odds_player1': 'Player 1 Odds',
            'odds_player2': 'Player 2 Odds',
            'recommended_bet_on': 'Recommended Bet On',
            'recommended_odds': 'Betting Odds',
            'recommended_probability': 'Model Probability',
            'recommended_expected_value': 'Expected Value',
            'recommended_prob_diff': 'Probability Difference',
            'match_id': 'Match ID',
            'player1_id': 'Player 1 ID',
            'player2_id': 'Player 2 ID',
            'status': 'Status'
        }, inplace=True)

        return df_filtered