% % writefile
train_skeleton.py

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, Dense, Flatten, Multiply, AdditiveAttention
from keras._tf_keras.keras.callbacks import EarlyStopping
import joblib
import boto3
import os


def process_financial_data(df, price_type, factor):
    if df.isnull().sum().any():
        df.fillna(method="ffill", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[price_type].values.reshape(-1, 1))

    X, y = [], []
    for i in range(factor, len(scaled_data)):
        X.append(scaled_data[i - factor:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


def train_model(data_path, model_name, output_bucket, output_prefix):
    # Load data from S3
    s3 = boto3.client('s3')
    local_file = '/tmp/data.csv'
    bucket, key = data_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_file)

    df = pd.read_csv(local_file)
    price_type = "Close"
    factor = 60

    X, y, scaler = process_financial_data(df, price_type, factor)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

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

    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=25,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    # Save model locally
    local_model_path = f'/tmp/{model_name}.h5'
    model.save(local_model_path)

    # Save scaler locally
    local_scaler_path = f'/tmp/{model_name}_scaler.joblib'
    joblib.dump(scaler, local_scaler_path)

    # Upload model and scaler to S3
    s3_model_path = f'{output_prefix}/{model_name}.h5'
    s3_scaler_path = f'{output_prefix}/{model_name}_scaler.joblib'
    s3.upload_file(local_model_path, output_bucket, s3_model_path)
    s3.upload_file(local_scaler_path, output_bucket, s3_scaler_path)

    print(f"Model and scaler saved to s3://{output_bucket}/{output_prefix}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--output-bucket', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path, args.model_name, args.output_bucket, args.output_prefix)