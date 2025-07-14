import pygsheets
import pandas as pd
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GoogleSheetsManager:
    def __init__(self, spreadsheet_name="Tennis Predictions"): # تم تغيير اسم جدول البيانات هنا
        self.spreadsheet_name = spreadsheet_name
        self.gc = self._authorize_google_sheets()
        self.sh = None
        self.live_predictions_ws = None
        self.backtesting_ws = None
        self.system_performance_ws = None

    def _authorize_google_sheets(self):
        try:
            # تأكد من أن المسار لملف الخدمة صحيح
            gc = pygsheets.authorize(service_file='darkechate-026c305d6750.json')
            logging.info("Google Sheets API authorized successfully.")
            return gc
        except Exception as e:
            logging.critical(f"Failed to authorize Google Sheets API. Make sure 'darkechate-026c305d6750.json' is correctly configured and in the root directory. Error: {e}")
            raise

    def create_or_get_sheets(self):
        try:
            self.sh = self.gc.open(self.spreadsheet_name)
            logging.info(f"Spreadsheet '{self.spreadsheet_name}' opened.")
        except pygsheets.SpreadsheetNotFound:
            logging.info(f"Spreadsheet '{self.spreadsheet_name}' not found. Creating a new one...")
            self.sh = self.gc.create(self.spreadsheet_name)
            self.sh.share('', role='writer', type='anyone') # تأكد أن هذا آمن لبيئة عملك
            logging.info(f"Spreadsheet '{self.spreadsheet_name}' created and shared.")

        # Define headers for Live Predictions
        # IMPORTANT: These headers MUST match the columns returned by model_predictor.predict() and filter_predictions()
        live_predictions_headers = [
            'Match ID', 'Prediction Date', 'Match Date', 'Match Time', 'Tournament',
            'Player 1', 'Player 2', 'Recommended Bet On', 'Betting Odds',
            'Model Probability', 'Expected Value', 'Probability Difference',
            'Player 1 Odds', 'Player 2 Odds', 'Player 1 ID', 'Player 2 ID', 'Status'
        ]

        try:
            self.live_predictions_ws = self.sh.worksheet('title', 'Live Predictions')
            logging.info("Worksheet 'Live Predictions' found.")
            # Verify headers and reset if mismatch
            current_headers = self.live_predictions_ws.get_values(start='A1', end='Z1')[0]
            if current_headers != live_predictions_headers:
                logging.warning("Live Predictions headers mismatch or not found. Resetting headers.")
                self.live_predictions_ws.clear()
                self.live_predictions_ws.set_dataframe(pd.DataFrame(columns=live_predictions_headers), start='A1', copy_head=True)
        except pygsheets.WorksheetNotFound:
            self.live_predictions_ws = self.sh.add_worksheet('Live Predictions')
            self.live_predictions_ws.set_dataframe(pd.DataFrame(columns=live_predictions_headers), start='A1', copy_head=True)
            logging.info("Worksheet 'Live Predictions' created and headers set.")

        # Define headers for Backtesting Results
        # IMPORTANT: These headers MUST match the cols_to_return in backtesting_logic.py
        backtesting_headers = [
            'Match ID', 'Prediction Date', 'Match Date', 'Match Time',
            'Tournament', 'Player 1', 'Player 2', 'Recommended Bet On',
            'Betting Odds', 'Model Probability', 'Expected Value',
            'Probability Difference', 'Actual Winner', 'Profit/Loss', 'Bet Amount',
            'Backtested Date'
        ]

        try:
            self.backtesting_ws = self.sh.worksheet('title', 'Backtesting Results')
            logging.info("Worksheet 'Backtesting Results' found.")
            # Verify headers and reset if mismatch
            current_headers = self.backtesting_ws.get_values(start='A1', end='Z1')[0]
            if current_headers != backtesting_headers:
                logging.warning("Backtesting Results headers mismatch or not found. Resetting headers.")
                self.backtesting_ws.clear()
                self.backtesting_ws.set_dataframe(pd.DataFrame(columns=backtesting_headers), start='A1', copy_head=True)
        except pygsheets.WorksheetNotFound:
            self.backtesting_ws = self.sh.add_worksheet('Backtesting Results')
            self.backtesting_ws.set_dataframe(pd.DataFrame(columns=backtesting_headers), start='A1', copy_head=True)
            logging.info("Worksheet 'Backtesting Results' created and headers set.")

        # Define headers for System Performance
        system_performance_headers = [
            'Date', 'Overall Profit/Loss', 'Total Bets', 'Total Wins', 'Total Losses',
            'Win Rate (%)', 'Average Odds', 'Highest Odds Win', 'Lowest Odds Loss'
        ]

        try:
            self.system_performance_ws = self.sh.worksheet('title', 'System Performance')
            logging.info("Worksheet 'System Performance' found.")
            # Verify headers and reset if mismatch
            current_headers = self.system_performance_ws.get_values(start='A1', end='Z1')[0]
            if current_headers != system_performance_headers:
                logging.warning("System Performance headers mismatch or not found. Resetting headers.")
                self.system_performance_ws.clear()
                self.system_performance_ws.set_dataframe(pd.DataFrame(columns=system_performance_headers), start='A1', copy_head=True)
        except pygsheets.WorksheetNotFound:
            self.system_performance_ws = self.sh.add_worksheet('System Performance')
            self.system_performance_ws.set_dataframe(pd.DataFrame(columns=system_performance_headers), start='A1', copy_head=True)
            logging.info("Worksheet 'System Performance' created and headers set.")

    def update_live_predictions(self, df_predictions):
        if self.live_predictions_ws is None:
            logging.error("Live Predictions worksheet not initialized.")
            return

        # Define the exact order of columns as per sheet headers
        cols_order = [
            'Match ID', 'Prediction Date', 'Match Date', 'Match Time', 'Tournament',
            'Player 1', 'Player 2', 'Recommended Bet On', 'Betting Odds',
            'Model Probability', 'Expected Value', 'Probability Difference',
            'Player 1 Odds', 'Player 2 Odds', 'Player 1 ID', 'Player 2 ID', 'Status'
        ]
        
        # Ensure all columns in cols_order are in df_predictions. If not, add them with None/NaN.
        for col in cols_order:
            if col not in df_predictions.columns:
                logging.warning(f"Column '{col}' not found in predictions DataFrame. Adding with default value.")
                df_predictions[col] = None # Using None for missing data to match pygsheets handling

        df_predictions_to_sheet = df_predictions[cols_order].copy()

        # Convert IDs to string to avoid scientific notation in sheets
        if 'Match ID' in df_predictions_to_sheet.columns:
            df_predictions_to_sheet['Match ID'] = df_predictions_to_sheet['Match ID'].astype(str)
        if 'Player 1 ID' in df_predictions_to_sheet.columns:
            df_predictions_to_sheet['Player 1 ID'] = df_predictions_to_sheet['Player 1 ID'].astype(str)
        if 'Player 2 ID' in df_predictions_to_sheet.columns:
            df_predictions_to_sheet['Player 2 ID'] = df_predictions_to_sheet['Player 2 ID'].astype(str)

        # Convert datetime.date objects to string for consistent sheet writing
        for col in ['Prediction Date', 'Match Date']:
            if col in df_predictions_to_sheet.columns and pd.api.types.is_datetime64_any_dtype(df_predictions_to_sheet[col]):
                df_predictions_to_sheet[col] = df_predictions_to_sheet[col].dt.strftime('%Y-%m-%d')
            elif col in df_predictions_to_sheet.columns: # If it's already date object, convert to string
                df_predictions_to_sheet[col] = df_predictions_to_sheet[col].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) else (str(x) if x is not None else ''))

        try:
            # Fetch existing data from the sheet
            existing_df = self.get_live_predictions_from_sheet()
            
            # Combine new predictions with existing ones, dropping duplicates based on 'Match ID'
            # Keep the new prediction if a match ID already exists (e.g., if odds updated)
            if not existing_df.empty:
                # Ensure 'Match ID' is string type in both for proper merging/deduplication
                existing_df['Match ID'] = existing_df['Match ID'].astype(str)
                df_predictions_to_sheet['Match ID'] = df_predictions_to_sheet['Match ID'].astype(str)
                
                combined_df = pd.concat([existing_df, df_predictions_to_sheet]).drop_duplicates(subset=['Match ID'], keep='last')
            else:
                combined_df = df_predictions_to_sheet

            # Write the combined DataFrame back to the sheet
            self.live_predictions_ws.clear(start='A2') # Clear data rows, keeping headers
            if not combined_df.empty:
                # Filter out the header row if it's accidentally in the data
                combined_df_no_headers = combined_df[~combined_df['Match ID'].isin(['Match ID', ''])] # Filter out rows that are headers
                if not combined_df_no_headers.empty:
                    self.live_predictions_ws.set_dataframe(combined_df_no_headers, start='A2', copy_head=False)
                    logging.info(f"Live predictions updated in Google Sheet. Total {len(combined_df_no_headers)} records.")
                else:
                    logging.info("No valid live predictions to update after combining and filtering headers.")
            else:
                logging.info("No new live predictions to update, cleared previous data.")
        except Exception as e:
            logging.error(f"Failed to update live predictions in Google Sheet: {e}", exc_info=True)


    def get_live_predictions_from_sheet(self):
        """
        Fetches all records from the 'Live Predictions' sheet.
        """
        if self.live_predictions_ws is None:
            logging.error("Live Predictions worksheet not initialized. Cannot fetch live predictions.")
            return pd.DataFrame()
        try:
            # Use get_as_df for easier DataFrame creation with headers
            df = self.live_predictions_ws.get_as_df(has_header=True)
            
            # Ensure 'Match ID' is string type
            if 'Match ID' in df.columns:
                df['Match ID'] = df['Match ID'].astype(str)
            # Convert date columns to date objects if they exist
            for col in ['Match Date', 'Prediction Date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            
            logging.info(f"Successfully retrieved {len(df)} records from Live Predictions sheet.")
            return df
        except Exception as e:
            logging.error(f"Error reading live predictions from Google Sheets: {e}", exc_info=True)
            return pd.DataFrame()


    def get_backtesting_results_from_sheet(self):
        if self.backtesting_ws is None:
            logging.error("Backtesting Results worksheet not initialized. Returning empty DataFrame.")
            return pd.DataFrame()
        try:
            df = self.backtesting_ws.get_as_df(has_header=True)
            
            # Ensure specific columns are handled as strings for IDs and converted for dates
            if 'Match ID' in df.columns:
                df['Match ID'] = df['Match ID'].astype(str)
            if 'Prediction Date' in df.columns:
                df['Prediction Date'] = pd.to_datetime(df['Prediction Date'], errors='coerce').dt.date
            if 'Match Date' in df.columns:
                df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce').dt.date
            if 'Backtested Date' in df.columns:
                df['Backtested Date'] = pd.to_datetime(df['Backtested Date'], errors='coerce').dt.date
                
            logging.info(f"Successfully retrieved {len(df)} backtesting records from sheet.")
            return df
        except Exception as e:
            logging.error(f"Failed to get backtesting results from Google Sheet: {e}", exc_info=True)
            return pd.DataFrame()

    def update_backtesting_results_full_sheet(self, df_results):
        """
        Clears the 'Backtesting Results' sheet and writes the entire DataFrame.
        This is crucial for updating existing backtested matches and preventing duplicates.
        """
        if self.backtesting_ws is None:
            logging.error("Backtesting Results worksheet not initialized. Cannot update full sheet.")
            return
        
        # Define the exact order of columns as per sheet headers (important for pygsheets)
        # These headers MUST match the backtesting_headers defined in create_or_get_sheets
        # and cols_to_return in backtesting_logic.py
        cols_order = [
            'Match ID', 'Prediction Date', 'Match Date', 'Match Time',
            'Tournament', 'Player 1', 'Player 2', 'Recommended Bet On',
            'Betting Odds', 'Model Probability', 'Expected Value',
            'Probability Difference', 'Actual Winner', 'Profit/Loss', 'Bet Amount',
            'Backtested Date'
        ]

        # Ensure all required columns are in the DataFrame and in the correct order
        for col in cols_order:
            if col not in df_results.columns:
                logging.warning(f"Column '{col}' not found in backtesting results DataFrame for full sheet update. Adding with None/NaN.")
                df_results[col] = None # Or np.nan for numerical, but None is safer for mixed types

        df_results_to_sheet = df_results[cols_order].copy()
        
        # Convert IDs to string
        if 'Match ID' in df_results_to_sheet.columns:
            df_results_to_sheet['Match ID'] = df_results_to_sheet['Match ID'].astype(str)
        
        # Convert datetime.date objects to string for consistent sheet writing
        for col in ['Prediction Date', 'Match Date', 'Backtested Date']:
            if col in df_results_to_sheet.columns and pd.api.types.is_datetime64_any_dtype(df_results_to_sheet[col]):
                df_results_to_sheet[col] = df_results_to_sheet[col].dt.strftime('%Y-%m-%d')
            elif col in df_results_to_sheet.columns: # If it's already date object, convert to string
                df_results_to_sheet[col] = df_results_to_sheet[col].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) else (str(x) if x is not None else ''))


        try:
            self.backtesting_ws.clear() # Clear the entire sheet, including headers
            if not df_results_to_sheet.empty:
                self.backtesting_ws.set_dataframe(df_results_to_sheet, start='A1', copy_head=True)
                logging.info(f"Backtesting Results sheet fully updated with {len(df_results_to_sheet)} records.")
            else:
                # If df is empty, just write headers
                self.backtesting_ws.set_dataframe(pd.DataFrame(columns=cols_order), start='A1', copy_head=True)
                logging.info("Backtesting Results sheet cleared (no results to write).")

        except Exception as e:
            logging.error(f"Error updating Backtesting Results sheet fully: {e}", exc_info=True)
            raise


    def update_system_performance(self, df_performance):
        if self.system_performance_ws is None:
            logging.error("System Performance worksheet not initialized.")
            return

        if df_performance.empty:
            logging.warning("No system performance data to update.")
            # Clear previous data if no new data to write, but keep headers
            try:
                self.system_performance_ws.clear(start='A2')
                logging.info("System Performance sheet data cleared (no new performance data).")
            except Exception as e:
                logging.error(f"Failed to clear system performance sheet: {e}", exc_info=True)
            return

        df_performance_to_sheet = df_performance.copy()
        
        # Ensure 'Date' column is formatted correctly
        if 'Date' in df_performance_to_sheet.columns:
            df_performance_to_sheet['Date'] = pd.to_datetime(df_performance_to_sheet['Date']).dt.strftime('%Y-%m-%d')

        try:
            # Clear data rows, keep headers
            self.system_performance_ws.clear(start='A2') 
            self.system_performance_ws.set_dataframe(df_performance_to_sheet, start='A2', copy_head=False)
            logging.info("System performance updated in Google Sheet.")
        except Exception as e:
            logging.error(f"Failed to update system performance in Google Sheet: {e}", exc_info=True)