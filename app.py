import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Load the pre-trained model
model = tf.keras.models.load_model('C:\\Users\\SUTIKSHAN\\Desktop\\myWork\\Stocks Prediction model\\Stocks Prediction Model.keras')

# Streamlit header
st.header("MarketMaven")
stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = datetime.today().strftime('%Y-%m-%d')  # Set end date to today's date

# Fetch stock data
data = yf.download(stock, start, end)

# Fetch the current exchange rate from USD to INR
exchange_rate_data = yf.download("INR=X", start=start, end=end)

# Check if the exchange rate data is not empty
if not exchange_rate_data.empty:
    exchange_rate = exchange_rate_data['Close'].iloc[-1]  # Get the last available exchange rate
else:
    st.warning("Exchange rate data is not available. Using a default rate of 80.0.")
    exchange_rate = 80.0  # Set a default exchange rate if data is unavailable

# Convert stock data to INR
data_inr = data * exchange_rate

# Display the stock data in INR
st.subheader("Stock Data (INR)")
st.write(data_inr)

data_inr = data['Close'] * exchange_rate

# Prepare training and test data
data_train = pd.DataFrame(data_inr[0:int(len(data_inr) * 0.80)])  # Train on 80% of the data
data_test = pd.DataFrame(data_inr[int(len(data_inr) * 0.80):])  # Test on the remaining 20%

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1, 1))  # Fit scaler on training data
data_test_scaled = scaler.transform(data_test.values.reshape(-1, 1))  # Transform test data

# Create X and y arrays for model prediction
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])  # Use the last 100 scaled prices as input
    y.append(data_test_scaled[i, 0])    # The target value is the next price

x, y = np.array(x), np.array(y)

# Predicting the data
predict = model.predict(x)

# Inverse transform the predicted values
predict_inverse = scaler.inverse_transform(predict.reshape(-1, 1))

# Convert predicted prices to INR
predict_inr = predict_inverse.flatten()  # Flatten the array to 1D for easier processing
y_inr = scaler.inverse_transform(y.reshape(-1, 1)).flatten()  # Inverse transform y as well

# Display graphs
st.subheader("Price vs MA50")
ma_50_days = pd.Series(data_inr).rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(data_inr, 'g', label="Close Price (INR)")
plt.legend()
st.pyplot(fig1)

st.subheader("Price vs MA50 vs MA100")
ma_100_days = pd.Series(data_inr).rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(ma_100_days, 'b', label="MA100")
plt.plot(data_inr, 'g', label="Close Price (INR)")
plt.legend()
st.pyplot(fig2)

st.subheader("Price vs MA100 vs MA200")
ma_200_days = pd.Series(data_inr).rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="MA100")
plt.plot(ma_200_days, 'b', label="MA200")
plt.plot(data_inr, 'g', label="Close Price (INR)")
plt.legend()
st.pyplot(fig3)

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict_inr, 'r', label="Predicted Price (INR)")
plt.plot(y_inr, 'g', label="Original Price (INR)")
plt.xlabel('Time')
plt.ylabel('Price (INR)')
plt.legend()
st.pyplot(fig4)

# Add Feature: Predict the Next Day's Price
last_100_days = data_inr[-100:]  # Get the last 100 closing prices
last_100_days_scaled = scaler.transform(last_100_days.values.reshape(-1, 1))  # Scale the last 100 days
last_100_days_scaled = last_100_days_scaled.reshape(1, 100, 1)  # Reshape for model input

# Predicting next day price
predicted_next_day_scaled = model.predict(last_100_days_scaled)

# Inverse transform to get the predicted price in the original scale
predicted_next_day_price = scaler.inverse_transform(predicted_next_day_scaled)

# Convert to INR
predicted_next_day_price_inr = predicted_next_day_price[0][0]  # Predicted price for tomorrow in INR

# Display the predicted next day price in INR
st.subheader(f"Predicted Tomorrow's Price for {stock} (INR)")
st.write(f"₹{predicted_next_day_price_inr:.2f}")
