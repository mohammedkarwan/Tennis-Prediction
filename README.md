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



