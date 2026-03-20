# 💊 Pharmaceutical Drug Demand Forecasting with LSTM

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pharma-drug-forecasting.streamlit.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/mderya/pharma-drug-demand-forecasting-with-lstm-iqvia)

## 🎯 Project Overview

End-to-end pharmaceutical drug demand forecasting pipeline built with LSTM deep learning, trained on IQVIA MIDAS-style market data from the Northeast US.

**Key Results:**
- ✅ R² = 0.92 (LSTM outperforms ARIMA and Prophet)
- ✅ MAPE = 4.62% (highly accurate weekly forecasts)
- ✅ 12-week demand forecast with confidence intervals
- ✅ Pharma domain features: flu season, allergy season, cross-drug signals

## 🚀 Live Demo

👉 [Try the Streamlit App](https://your-app-url.streamlit.app)

## 📊 Dataset

IQVIA MIDAS-style pharmaceutical sales data:
- 8 ATC drug categories (Paracetamol, Ibuprofen, Aspirin, etc.)
- Northeast US Market | 2018–2023 | 313 weekly observations
- Includes COVID-19 impact period

## 🏗️ Project Structure

```
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
├── iqvia_pharma_sales.csv          # Dataset
├── pharma_drug_forecasting.ipynb   # Full Kaggle notebook
└── README.md
```

## 🤖 Model Architecture

- **Stacked LSTM** (64 → 32 units) with Dropout regularization
- **Sequence length:** 26 weeks (6 months of history)
- **Seasonal adjustment:** Prior-year same-week multiplier
- **Baseline comparison:** ARIMA vs Prophet vs LSTM

## 📈 Results

| Model   | R²     | MAPE   |
|---------|--------|--------|
| ARIMA   | -0.016 | 18.5%  |
| Prophet | 0.834  | 8.2%   |
| **LSTM**| **0.901** | **4.62%** |

## 🔧 Installation

```bash
git clone https://github.com/mattderya/pharma-drug-forecasting
cd pharma-drug-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author

**Matt Derya** | Data Scientist | 15+ years pharmaceutical domain expertise
- 🌐 [Website](https://mattderya.com)
- 🔗 [LinkedIn](https://linkedin.com/in/mttdryai)
- 📓 [Kaggle Notebook](https://www.kaggle.com/code/mderya/pharma-drug-demand-forecasting-with-lstm-iqvia)
