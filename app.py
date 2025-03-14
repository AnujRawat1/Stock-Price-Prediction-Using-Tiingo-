import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
from tiingo import TiingoClient
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_path = r'D:\Python\Projects\Projects\Stock Price Prediction\Stock Predictions Model.keras'
model = load_model(model_path)

# Streamlit UI
st.header('📈 Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol:', 'GOOG')

# Define start and end date
start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d")

# Tiingo API configuration
key = '2e48839e3973061bffab134187746033037381fb'
config = {'api_key': key, 'session': True}
client = TiingoClient(config)

# Function to fetch stock data
def generate_data(stock_name, start_date, client):
    try:
        data = client.get_ticker_price(
            stock_name, fmt='json', 
            startDate=start_date, 
            endDate=date.today().strftime("%Y-%m-%d"), 
            frequency='daily'
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_csv(f'{stock_name}.csv', index=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch stock data
data = generate_data(stock_name=stock, start_date=start, client=client)

if data is not None and not data.empty:
    # Convert date column
    data['date'] = pd.to_datetime(data['date']).dt.date

    # Display stock data
    st.subheader('📊 Stock Data Preview')
    st.write(data.tail(7))

    # Extract closing prices
    df1 = data[['close']]
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    df1_scaled = scaler.fit_transform(df1)

    # Prepare input for prediction (last 100 days of data)
    last_100_days = df1_scaled[-100:].reshape(1, 100, 1)

    # Predict the next 30 days
    future_predictions = []
    future_dates = [date.today() + timedelta(days=i) for i in range(1, 31)]

    for _ in range(30):
        predicted_price = model.predict(last_100_days)
        future_predictions.append(predicted_price[0, 0])
        
        # Update last_100_days with new prediction
        last_100_days = np.append(last_100_days[:, 1:, :], [[[predicted_price[0, 0]]]], axis=1)

    # Inverse transform the predictions to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plot original stock prices
    st.subheader('📈 Stock Price Trend')
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(df1.index, df1, color='#4169E1', label='Closing Price')
    plt.xlabel('Days')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'Stock Price Trend for {stock}')
    plt.legend()
    st.pyplot(fig1)

    # Plot future predictions
    st.subheader('📉 30-Day Stock Price Forecast')
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(future_dates, future_predictions, color='red', marker='o', linestyle='dashed', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.xticks(rotation=45)
    plt.title(f'Predicted Stock Prices for Next 30 Days ({stock})')
    plt.legend()
    st.pyplot(fig2)

else:
    st.warning("⚠ No data available for the selected stock.")
