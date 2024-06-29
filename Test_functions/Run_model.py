import tensorflow as tf
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Download the model file
model_path = hf_hub_download(repo_id="Finforbes/Stock_predictor", filename="HDFCBANK.h5")

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the input data
input_path = 'data/HDFCBANK.csv'
data = pd.read_csv(input_path)

# Assuming 'Close' is the price_type you want to use
price_type = 'Close'
closing_prices = data[price_type].values

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scaling the data
scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

# Assuming factor is 60 (as in your original code)
factor = 60

# Reshaping the data for the model
X_latest = np.array([scaled_data[-factor:].reshape(factor)])
X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

# Predicting the next day
predicted_stock_price = model.predict(X_latest)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Predicting the next 20 days iteratively
predicted_prices = []
current_batch = scaled_data[-factor:].reshape(1, factor, 1)

for i in range(40):
    next_prediction = model.predict(current_batch)
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
    predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

print(predicted_prices)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(len(closing_prices)), closing_prices, label='Historical Prices')
plt.plot(range(len(closing_prices), len(closing_prices) + 40), predicted_prices, label='Predicted Prices', color='orange')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()