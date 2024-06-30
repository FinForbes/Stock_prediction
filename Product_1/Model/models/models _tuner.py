import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input,
    LSTM,
    Dense,
    Flatten,
    Multiply,
    AdditiveAttention,
    Dropout
)
from keras._tf_keras.keras.callbacks import EarlyStopping
import keras_tuner as kt


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
        X.append(scaled_data[i - factor:i, 0])
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


def build_model(hp):
    input_layer = Input(shape=(60, 1))

    # Tune the number of LSTM layers and units
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        if i == 0:
            x = LSTM(
                units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                return_sequences=True if i < hp.Int('num_lstm_layers', 1, 3) - 1 else False
            )(input_layer)
        else:
            x = LSTM(
                units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                return_sequences=True if i < hp.Int('num_lstm_layers', 1, 3) - 1 else False
            )(x)

        # Add dropout after each LSTM layer
        x = Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1))(x)

    # Attention mechanism
    if hp.Boolean("use_attention"):
        query = Dense(hp.Int('attention_dim', 32, 128, step=32))(x)
        value = Dense(hp.Int('attention_dim', 32, 128, step=32))(x)
        attention_out = AdditiveAttention()([query, value])
        x = Multiply()([x, attention_out])

    x = Flatten()(x)

    # Tune the number of dense layers
    for i in range(hp.Int('num_dense_layers', 0, 2)):
        x = Dense(
            units=hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16),
            activation='relu'
        )(x)
        x = Dropout(hp.Float(f'dense_dropout_{i}', 0, 0.5, step=0.1))(x)

    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Tune the learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    return model


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

        # Create a tuner
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory='my_dir',
            project_name='stock_prediction'
        )

        # Perform the search
        tuner.search(X_train, y_train, epochs=100, validation_split=0.2,
                     callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]

        # Print the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
        The hyperparameter search is complete. The optimal number of LSTM layers is {best_hps.get('num_lstm_layers')} and the optimal learning rate is {best_hps.get('learning_rate')}.
        """)

        # Train the model with the best hyperparameters
        history = best_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)]
        )

        model_name = filename.split(".")[0]

        # Save the best model
        best_model.save(f"../../Cloud/models/{model_name}_best.h5")
        file_processed += 1
        print(f"Model {model_name}_best.h5 saved successfully.")

        if file_processed >= 3:
            break