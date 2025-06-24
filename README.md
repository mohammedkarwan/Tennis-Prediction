# 🎾 Tennis Match Prediction – Value Betting with Machine Learning

An end-to-end machine learning project that predicts the outcomes of upcoming tennis matches and identifies value bets using live data from a sports API.

---

## 📌 Objective
To develop an intelligent betting assistant that uses a trained machine learning model to evaluate match odds and identify high-value betting opportunities.

---

## 🧠 How It Works

1. **Live Data Fetching**  
   Retrieves upcoming tennis matches using a sports API (`api-tennis.com`), including players, odds, and match metadata.

2. **Feature Engineering**  
   - Calculates derived features such as `log_odds`, `inverse_odds`, and `risk_index`.  
   - Encodes match level data for model compatibility.

3. **Prediction Model**  
   Loads a pre-trained classification model (`Tennis-Prediction.joblib`) and outputs win probability, expected value (EV), and suggested prediction.

4. **Value Bet Filtering**  
   Highlights only those bets where:
   - Win probability ≥ 60%  
   - Expected value ≥ 15%  

5. **Result Exporting**  
   Saves high-value betting opportunities to a CSV file for further use.

---

## 🧰 Tools & Technologies

- Python, NumPy, Pandas
- Scikit-learn (for model and preprocessing)
- Joblib (model serialization)
- Requests (API handling)
- Google Colab / Jupyter Notebook

---

## ✅ Sample Output

📅 High-Value Upcoming Matches:
🗓 2025-06-25 | Novak Djokovic vs Carlos Alcaraz
🎯 Prediction: Novak Djokovic to win match
📈 Prob: 71.24% | EV: 22.37% | Odds: 1.85

---

## 🚀 How to Run

1. Upload your trained model file: Tennis-Prediction.joblib

2. Set your API key from api-tennis.com

3. Run the script in a Google Colab or local environment

4. Check the console for high-value predictions and download the CSV file

---
## 📊 Outcome

The project effectively detects profitable betting opportunities in real time, showing the power of machine learning in sports analytics


