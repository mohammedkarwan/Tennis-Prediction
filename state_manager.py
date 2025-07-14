import json
import os
import logging
from datetime import date, datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SystemStateManager:
    def __init__(self, state_file='system_state.json'):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Convert date strings back to date objects
                    if 'last_run_date' in state and state['last_run_date']:
                        state['last_run_date'] = datetime.strptime(state['last_run_date'], '%Y-%m-%d').date()
                    if 'last_retrain_date' in state and state['last_retrain_date']:
                        state['last_retrain_date'] = datetime.strptime(state['last_retrain_date'], '%Y-%m-%d').date()
                    logging.info("System state loaded successfully.")
                    return state
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from state file {self.state_file}: {e}")
            except Exception as e:
                logging.error(f"Error loading system state from {self.state_file}: {e}")
        logging.info("No existing system state found or error loading. Starting with default state.")
        return {'last_run_date': None, 'last_retrain_date': None}

    def _save_state(self):
        try:
            state_to_save = self.state.copy()
            # Convert date objects to strings for JSON serialization
            if state_to_save['last_run_date']:
                state_to_save['last_run_date'] = state_to_save['last_run_date'].strftime('%Y-%m-%d')
            if state_to_save['last_retrain_date']:
                state_to_save['last_retrain_date'] = state_to_save['last_retrain_date'].strftime('%Y-%m-%d')

            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            logging.info("System state saved successfully.")
        except Exception as e:
            logging.error(f"Error saving system state to {self.state_file}: {e}")

    def get_last_run_date(self):
        return self.state.get('last_run_date')

    def set_last_run_date(self, run_date: date):
        self.state['last_run_date'] = run_date
        self._save_state()

    def get_last_retrain_date(self):
        return self.state.get('last_retrain_date')

    def set_last_retrain_date(self, retrain_date: date):
        self.state['last_retrain_date'] = retrain_date
        self._save_state()