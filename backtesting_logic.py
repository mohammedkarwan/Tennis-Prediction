import pandas as pd
import logging
from datetime import date
from config import BET_AMOUNT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestingLogic:
    def __init__(self):
        pass

    def perform_backtesting(self, finished_matches_df, existing_predictions_df):
        if finished_matches_df.empty or existing_predictions_df.empty:
            logging.info("Not enough data for backtesting: either no new finished matches or no existing predictions.")
            return pd.DataFrame()

        # Ensure 'Match ID' is consistent type
        finished_matches_df['match_id'] = finished_matches_df['match_id'].astype(str)
        existing_predictions_df['Match ID'] = existing_predictions_df['Match ID'].astype(str)

        # Merge predictions with actual finished match results
        # We need to link the prediction made on day X to the match that finished on day Y
        # Match ID is the key
        merged_df = pd.merge(
            existing_predictions_df,
            finished_matches_df[['match_id', 'winner_id', 'player1_id', 'player2_id']],
            left_on='Match ID',
            right_on='match_id',
            how='inner' # Only consider predictions for matches that have now finished
        )

        if merged_df.empty:
            logging.info("No common matches between existing predictions and newly finished matches for backtesting.")
            return pd.DataFrame()

        # Identify actual winner based on winner_id
        merged_df['Actual Winner'] = merged_df.apply(lambda row: \
            row['Player 1'] if str(row['winner_id']) == str(row['player1_id']) else \
            (row['Player 2'] if str(row['winner_id']) == str(row['player2_id']) else 'Unknown'), axis=1
        )
        
        # Calculate Profit/Loss
        merged_df['Profit/Loss'] = 0.0
        merged_df['Bet Amount'] = BET_AMOUNT

        for index, row in merged_df.iterrows():
            if row['Recommended Bet On'] == row['Actual Winner']:
                # Win
                profit = (row['Betting Odds'] * row['Bet Amount']) - row['Bet Amount']
                merged_df.at[index, 'Profit/Loss'] = profit
            elif row['Actual Winner'] == 'Unknown':
                # Match result not clear (e.g., retired, walkover, or API didn't provide winner)
                merged_df.at[index, 'Profit/Loss'] = 0 # No profit/loss until clear outcome
            else:
                # Loss
                merged_df.at[index, 'Profit/Loss'] = -row['Bet Amount']

        # Filter out predictions that have already been backtested (if 'Profit/Loss' is not 0 for example)
        # Or, filter for unique Match IDs that are new to backtesting
        # For simplicity, we assume that any match in merged_df that was previously 0.0 P/L is now updated.
        # A more robust solution might require a "backtested" flag in the sheet.
        
        # For now, we only want to return new or updated backtesting results.
        # This will append all calculated results, which means duplicate Match IDs might occur if rerun on same finished matches.
        # A better approach would be to fetch existing sheet, update in-memory, then overwrite/update sheet.
        # For this simplified append, we'll append all. It's up to GoogleSheetsManager to handle uniqueness.

        # Prepare for appending to sheet
        cols_to_return = [
            'Match ID', 'Prediction Date', 'Match Date', 'Match Time',
            'Tournament', 'Player 1', 'Player 2', 'Recommended Bet On',
            'Betting Odds', 'Model Probability', 'Expected Value',
            'Probability Difference', 'Actual Winner', 'Profit/Loss', 'Bet Amount'
        ]
        
        # Ensure all columns exist before selecting
        for col in cols_to_return:
            if col not in merged_df.columns:
                merged_df[col] = None # Add missing column with None/NaN

        return merged_df[cols_to_return]


    def calculate_system_performance(self, all_backtesting_results_df):
        if all_backtesting_results_df.empty:
            logging.info("No backtesting results to calculate system performance.")
            return pd.DataFrame(columns=[
                'Date', 'Overall Profit/Loss', 'Total Bets', 'Total Wins', 'Total Losses',
                'Win Rate (%)', 'Average Odds', 'Highest Odds Win', 'Lowest Odds Loss'
            ])

        # Ensure Profit/Loss is numeric
        all_backtesting_results_df['Profit/Loss'] = pd.to_numeric(all_backtesting_results_df['Profit/Loss'], errors='coerce').fillna(0)
        all_backtesting_results_df['Betting Odds'] = pd.to_numeric(all_backtesting_results_df['Betting Odds'], errors='coerce').fillna(1.0)

        total_profit_loss = all_backtesting_results_df['Profit/Loss'].sum()
        total_bets = len(all_backtesting_results_df)
        total_wins = (all_backtesting_results_df['Profit/Loss'] > 0).sum()
        total_losses = (all_backtesting_results_df['Profit/Loss'] < 0).sum()
        
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0

        # Filter for winning bets to calculate average odds for wins
        winning_bets = all_backtesting_results_df[all_backtesting_results_df['Profit/Loss'] > 0]
        average_odds = winning_bets['Betting Odds'].mean() if len(winning_bets) > 0 else 0

        # Highest odds win and lowest odds loss
        highest_odds_win = winning_bets['Betting Odds'].max() if len(winning_bets) > 0 else 0
        
        losing_bets = all_backtesting_results_df[all_backtesting_results_df['Profit/Loss'] < 0]
        lowest_odds_loss = losing_bets['Betting Odds'].min() if len(losing_bets) > 0 else 0


        performance_data = {
            'Date': [date.today().strftime('%Y-%m-%d')],
            'Overall Profit/Loss': [f"{total_profit_loss:.2f}"],
            'Total Bets': [total_bets],
            'Total Wins': [total_wins],
            'Total Losses': [total_losses],
            'Win Rate (%)': [f"{win_rate:.2f}%"],
            'Average Odds': [f"{average_odds:.2f}"],
            'Highest Odds Win': [f"{highest_odds_win:.2f}"],
            'Lowest Odds Loss': [f"{lowest_odds_loss:.2f}"]
        }
        
        df_performance = pd.DataFrame(performance_data)
        return df_performance