# 🎾 Tennis Match Prediction — Value Betting with Machine Learning

An end-to-end machine learning system designed to predict the outcomes of upcoming tennis matches and detect **value bets** based on real-time data from a sports API.

---

## 🎯 Project Goal

To build an intelligent betting assistant that evaluates match odds using a trained ML model and identifies high-value betting opportunities with strong profit potential.

---

## 🧠 How It Works

1. **📡 Real-Time Match Data**  
   Retrieves live tennis match information (players, odds, match level) from [api-tennis.com](https://api-tennis.com/).

2. **🛠 Feature Engineering**  
   - Generates key features: `log_odds`, `inverse_odds`, `risk_index`.  
   - Processes and encodes data for ML compatibility.

3. **🔮 Prediction Engine**  
   - Loads a pre-trained model (`Tennis-Prediction.joblib`) to compute win probabilities and expected value (EV).  
   - Predicts match outcomes and betting value.

4. **🎯 Value Bet Filtering**  
   Filters predictions based on configurable thresholds:
   - Win probability ≥ 60%  
   - Expected value ≥ 15%

5. **📥 Exporting Results**  
   Saves profitable match predictions to a downloadable CSV file.

---

## ⚙️ Technologies Used

- **Languages & Libraries**: Python, NumPy, Pandas  
- **ML & Modeling**: Scikit-learn, Joblib  
- **Data Access**: Requests (API handling)  
- **Environment**: Google Colab / Jupyter Notebook

---

## 📈 Example Output

📅 2025-06-25 | Novak Djokovic vs Carlos Alcaraz
🎯 Prediction: Novak Djokovic to win
🔢 Probability: 71.24%
💰 EV: 22.37%
📊 Odds: 1.85

---

## 🚀 How to Use

1. Upload your trained model file: `Tennis-Prediction.joblib`
2. Set your API key from [api-tennis.com](https://api-tennis.com/)
3. Run the notebook or Python script
4. View predictions and download the CSV output

---

## 💡 Results

This project demonstrates how machine learning can uncover **profitable betting opportunities** in real time, using a systematic and data-driven approach.
