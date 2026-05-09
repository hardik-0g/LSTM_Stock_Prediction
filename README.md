# LSTM Stock Price Prediction 📈

A deep learning-based stock market prediction web application built using **LSTM (Long Short-Term Memory)** neural networks, **TensorFlow/Keras**, and **Streamlit**. The application analyzes historical stock market data and predicts future stock price trends using sequential time-series forecasting techniques.

## 🚀 Live Demo

https://lstm-stock-prediction-pxk3.onrender.com

---

## 📌 Overview

This project uses an LSTM-based recurrent neural network trained on historical stock price data to predict future closing prices. The model learns temporal dependencies and sequential market patterns from previous stock movements.

The application fetches real-time stock data using Yahoo Finance and provides interactive visualizations for analysis and prediction.

---

## 🧠 Deep Learning Concepts Used

- Long Short-Term Memory (LSTM)
- Recurrent Neural Networks (RNN)
- Time Series Forecasting
- Sequential Data Modeling
- Sliding Window Technique
- Data Normalization using MinMaxScaler

---

## ⚙️ Features

- Real-time stock data fetching
- LSTM-based stock price prediction
- Historical data visualization
- 100-Day & 200-Day Moving Average graphs
- Predicted vs Actual price comparison
- Interactive Streamlit UI
- Cloud deployment on Render

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core Programming |
| TensorFlow / Keras | Deep Learning |
| Streamlit | Web Interface |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Matplotlib | Visualization |
| Scikit-learn | Data Scaling |
| Yahoo Finance API | Stock Data |

---

## 📂 Project Structure

```bash
LSTM_Stock_Prediction/
│
├── app.py
├── Stock Predictions Model.keras
├── requirements.txt
├── runtime.txt
├── .python-version
└── README.md
```

---

## 🔄 Workflow

```text
User Input → Yahoo Finance API → Data Preprocessing →
Normalization → Sequence Generation →
LSTM Prediction → Visualization
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/hardik-0g/LSTM_Stock_Prediction.git
```

Move into the project directory:

```bash
cd LSTM_Stock_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 Example Stocks

You can test predictions using:
- AAPL
- TSLA
- NVDA
- GOOGL
- MSFT

---

## 🌐 Deployment

The application is deployed on Render using:
- Python 3.11
- Streamlit
- TensorFlow CPU

---

## 📌 Future Improvements

- Multi-stock forecasting
- Transformer-based prediction models
- Sentiment analysis integration
- Real-time streaming data
- Advanced technical indicators

---

## ⚠️ Disclaimer

This project is built for educational and research purposes only. Stock market predictions are uncertain and should not be considered financial advice.

---

## 👨‍💻 Author

**Hardik Jaiswal**

GitHub:  
https://github.com/hardik-0g

**Kalpit Yadav**

GitHub:
https://github.com/kalpit71


---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.