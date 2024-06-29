import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input,
    LSTM,
    Dense,
    Flatten,
    Multiply,
    AdditiveAttention,
)
from keras._tf_keras.keras.callbacks import EarlyStopping


# Function to process financial data
def process_financial_data(df, price_type, factor):
    # Check for missing values and fill them
    if df.isnull().sum().any():
        df.fillna(method="ffill", inplace=True)

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[price_type].values.reshape(-1, 1))

    X, y, dates = [], [], []
    for i in range(factor, len(scaled_data)):
        X.append(scaled_data[i-factor:i, 0])
        y.append(scaled_data[i, 0])
        dates.append(df.index[i])  # Assuming the index is the date

    X, y, dates = np.array(X), np.array(y), np.array(dates)
    return X, y, dates, scaler


# Function to process CSV files in a folder
def process_csv_files(folder_path, filename, price_type, factor, max_files=3):

    data_list = []
    labels = []
    scalers = []

    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        X, y, dates, scaler = process_financial_data(df, price_type, factor)
        data_list.append((X, y, dates))
        scalers.append(scaler)

        label = filename.split(".")[0]  # Extract label from filename
        labels.append(label)

    return data_list, labels, scalers


if __name__ == "__main__":
    # Define your parameters here
    folder_path = "../../Test_functions/data"
    price_type = "Close"  # e.g., "Close", "Open", "High", "Low"
    factor = 60  # This can be the number of past days you consider for your model
    file_processed = 0
    for filename in os.listdir(folder_path):
        print(filename)
        # Process CSV files
        data_list, labels, scalers = process_csv_files(
            folder_path, filename, price_type, factor
        )
        print(data_list)

        if not data_list or all(len(x[0]) == 0 for x in data_list):
            print(f"No valid data found in {filename}. Skipping...")
            continue

        # Convert data_list to numpy arrays
        X_list, y_list, dates_list = zip(*data_list)
        X_array = np.concatenate(X_list, axis=0)
        y_array = np.concatenate(y_list, axis=0)
        dates_array = np.concatenate(dates_list, axis=0)

        if len(X_array) == 0:
            print(f"No valid data found in {filename} after processing. Skipping...")
            continue

        # Split the data into training and testing sets
        train_size = int(len(X_array) * 0.8)
        X_train, X_test = X_array[:train_size], X_array[train_size:]
        y_train, y_test = y_array[:train_size], y_array[train_size:]
        dates_train, dates_test = dates_array[:train_size], dates_array[train_size:]
        print(X_train.shape, y_train.shape)

        # Reshape for LSTM input
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Define the model
        input_layer = Input(shape=(X_train.shape[1], 1))
        lstm_out = LSTM(50, return_sequences=True)(input_layer)
        lstm_out = LSTM(50, return_sequences=True)(lstm_out)

        # Attention mechanism
        query = Dense(50)(lstm_out)
        value = Dense(50)(lstm_out)
        attention_out = AdditiveAttention()([query, value])

        # Combine LSTM output with attention output
        multiply_layer = Multiply()([lstm_out, attention_out])

        # Flatten and output
        flatten_layer = Flatten()(multiply_layer)
        output_layer = Dense(1)(flatten_layer)

        # Compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.summary()

        # Train the model
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=25,
            validation_split=0.2,
            callbacks=[early_stopping],
        )

        model_name = filename.split(".")[0]

        # Save the model
        model.save(f"../../Cloud/models/{model_name}.h5")
        file_processed += 1
        print(f"Model {filename}.h5 saved successfully.")

        if file_processed >= 3:
            break
