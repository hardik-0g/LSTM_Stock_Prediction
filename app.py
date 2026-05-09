import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: white;
}

.metric-container {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

model = load_model('Stock Predictions Model.keras')

st.title("📈 AI Powered Stock Market Predictor")

st.markdown("""
### Predict Stock Prices using Deep Learning & LSTM Neural Networks
""")

st.sidebar.header("⚙️ Settings")

stock = st.sidebar.text_input(
    'Enter Stock Symbol',
    'GOOG'
)

start = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2012-01-01")
)

end = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("2022-12-31")
)

with st.spinner('Fetching Stock Data...'):
    data = yf.download(stock, start=start, end=end)

st.subheader('📊 Stock Data')

st.dataframe(data.tail())

col1, col2, col3 = st.columns(3)

latest_price = round(data['Close'].iloc[-1], 2)
highest_price = round(data['High'].max(), 2)
lowest_price = round(data['Low'].min(), 2)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Current Price</h3>
        <h2>${latest_price}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Highest Price</h3>
        <h2>${highest_price}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <h3>Lowest Price</h3>
        <h2>${lowest_price}</h2>
    </div>
    """, unsafe_allow_html=True)

data_train = pd.DataFrame(
    data.Close[0:int(len(data) * 0.80)]
)

data_test = pd.DataFrame(
    data.Close[int(len(data) * 0.80):]
)

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)

data_test = pd.concat(
    [past_100_days, data_test],
    ignore_index=True
)

data_test_scale = scaler.fit_transform(data_test)

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

st.subheader('📉 Price vs MA50')

fig1 = plt.figure(figsize=(12,6))

plt.plot(ma_50_days, label='MA50')
plt.plot(data.Close, label='Close Price')

plt.legend()
plt.grid(True)

st.pyplot(fig1)

st.subheader('📉 Price vs MA50 vs MA100')

fig2 = plt.figure(figsize=(12,6))

plt.plot(ma_50_days, label='MA50')
plt.plot(ma_100_days, label='MA100')
plt.plot(data.Close, label='Close Price')

plt.legend()
plt.grid(True)

st.pyplot(fig2)

st.subheader('📉 Price vs MA100 vs MA200')

fig3 = plt.figure(figsize=(12,6))

plt.plot(ma_100_days, label='MA100')
plt.plot(ma_200_days, label='MA200')
plt.plot(data.Close, label='Close Price')

plt.legend()
plt.grid(True)

st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('🤖 Original Price vs Predicted Price')

fig4 = plt.figure(figsize=(12,6))

plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')

plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
plt.grid(True)

st.pyplot(fig4)

st.subheader("🔮 Next Day Prediction")

last_100 = data_test_scale[-100:]
last_100 = np.reshape(last_100, (1, 100, 1))

next_day = model.predict(last_100)

next_day_price = next_day[0][0] * scale[0]

st.success(f"Predicted Next Day Price: ${next_day_price:.2f}")

st.markdown("")