# 🎾 Tennis Betting Prediction System

## 📌 Filtering Thresholds

- **Expected Value**: `>= 0.05`
- **Probability Difference**: `>= 0.03`

---

## 📊 Google Sheets Management

### **Live Predictions**
Filtered predictions for ongoing (non-finished) matches are uploaded to the **Live Predictions** sheet in Google Sheets. This sheet is updated frequently to reflect the most recent forecasts.

### **Backtesting Results**
Evaluates past predictions against the actual match outcomes. The sheet is updated with profit/loss data for each bet.

### **System Performance**
Overall performance metrics are calculated and stored in the **System Performance** sheet in Google Sheets.

---

## 📈 Backtesting & Performance Tracking

The system reviews historical predictions and labels each as a win or loss once the match concludes.

- Profit/Loss is calculated based on a default $10 bet per prediction.
- Key performance indicators tracked:
  - **Overall Profit/Loss**
  - **Total Bets**
  - **Total Wins**
  - **Total Losses**
  - **Win Rate (%)**
  - **Average Odds**
  - **Highest Odds Win**
  - **Lowest Odds Loss**

---

## 🔄 Automated Daily Run

The system is designed to run automatically once per day (or as configured), ensuring timely updates for predictions, data, and performance metrics.

---

## 📂 Project Structure

.
├── config.py                 # Configuration (API key, model paths, filter thresholds)
├── main_runner.py            # Main entry point, coordinates all modules
├── api_fetcher.py            # Fetches data from api-tennis.com
├── model_predictor.py        # Model loading/training, preprocessing, prediction, filtering
├── google_sheets_manager.py  # Google Sheets read/write functionality
├── backtesting_logic.py      # Backtesting logic and performance calculations
├── state_manager.py          # Manages system state (e.g., last run/train date)
├── best_tennis_model.pkl     # (Generated) Trained model file
├── scaler.pkl                # (Generated) Scaler used in preprocessing
└── darkechate-026c305d6750.json # Google Cloud Service Account credentials

---

## ⚙️ Technologies Used

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Requests, pytz, pygsheets  
- **Machine Learning**: Scikit-learn (classification models, preprocessing)  
- **API Integration**: Requests (for tennis data API)  
- **Spreadsheet Integration**: pygsheets (Google Sheets API)  
- **Environment**: Standalone Python script (ideal for scheduled daily runs)

---

## 🚀 Setup & Usage

### 1. Prerequisites

Ensure Python 3.8+ is installed. Then install dependencies:

```bash
pip install pandas pytz requests scikit-learn pygsheets
2. Google Sheets API Setup

1. Create a Google Cloud Project:
-Visit Google Cloud Console
-Create a new project or select an existing one.

2.Enable APIs:
-Go to APIs & Services > Enabled APIs & Services
-Enable both Google Sheets API and Google Drive API

3.Create a Service Account:
-Go to IAM & Admin > Service Accounts
-Click Create Service Account, give it a name, then click Done
-Click on the created account email > Keys > Add Key > Create new key > JSON
-A .json file will be downloaded (e.g., darkechate-026c305d6750.json). Save it securely.

4.Share the Google Sheet:
-Open the target Google Sheet
-Click Share
-Paste the service account email (...@developer.gserviceaccount.com)
-Grant Editor access and click Share

3. Configure config.py
Create a config.py file in your project directory:
# config.py

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

MODEL_PATH = "best_tennis_model.pkl"
SCALER_PATH = "scaler.pkl"
SAVE_MODEL_DIR = "."

RETRAIN_DAYS_LOOKBACK = 30
MIN_HISTORICAL_DAYS_FOR_TRAINING = 365

BET_AMOUNT = 10.0

MIN_MODEL_PROBABILITY = 0.60
MIN_MODEL_EXPECTED_VALUE = 0.05
MIN_PROBABILITY_DIFFERENCE = 0.03

4. Place Google Credentials File
Move the downloaded darkechate-026c305d6750.json file into your project root directory.

5. Run the System
From the command line, navigate to your project folder and run:
python main_runner.py

6. Automate with a Scheduler (Recommended)
-Use Task Scheduler:
-Open Task Scheduler > Create Basic Task
-Set trigger (e.g., daily)
-Action: Start a Program → python.exe
-Arguments: full path to main_runner.py
-Start in: folder path of your project
On Linux/macOS:
-Use Cron Jobs:
-Edit crontab with crontab -e
-Add a line for daily run at 3am, for example:
0 3 * * * /usr/bin/python3 /path/to/your/project/main_runner.py >> /path/to/your/project/cron.log 2>&1

---

📈 Key Performance Indicators (KPIs)
The system tracks and updates the following metrics in System Performance:

-Date
-Overall Profit/Loss
-Total Bets
-Total Wins
-Total Losses
-Win Rate (%)
-Average Odds
-Highest Odds Win
-Lowest Odds Loss

---

💡 Outcome

This project showcases how machine learning can identify profitable sports betting opportunities in real time using a structured, data-driven approach, with fully automated performance tracking.








