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


