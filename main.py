# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Load data from CSV file
csv_file_path = 'production_data_with_total_stock_and_sales.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Assuming 'Year', 'Months to This Year', 'Units Produced', 'Old Stocks', and 'Total Stock' are the columns in your CSV
X = df[['Year', 'Months to This Year', 'Units Produced', 'Old Stocks']]
y = df['Sales']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the Feedforward Neural Network (FNN) model
def build_model(optimizer='adam', neurons_layer1=64, neurons_layer2=32):
    model = Sequential()
    model.add(Dense(neurons_layer1, input_dim=5, activation='relu'))  # Update input_dim to 5
    model.add(Dense(neurons_layer2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create a KerasRegressor with the build_model function
keras_regressor = KerasRegressor(build_fn=build_model, epochs=100, batch_size=2, verbose=0)

# Define hyperparameters to tune
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'neurons_layer1': [32, 64, 128],
    'neurons_layer2': [16, 32, 64]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_result.best_params_
print(f'Best Hyperparameters: {best_params}')

# Use the best model for predictions
best_model = grid_result.best_estimator_

# Make predictions on new data
new_data = pd.DataFrame({
    'Year': [2026],  # Replace with the desired year
    'Months to This Year': [1],  # Replace with the desired month
    'Units Produced': [500],
    'Old Stocks': [-14],

})

# Standardize the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions
new_predictions = best_model.predict(new_data_scaled)

# Reverse the scaling for the predictions
new_predictions_inverse = scaler.inverse_transform(new_predictions.reshape(-1, 1)).flatten()

print(f'Predicted Sales for New Data: {new_predictions_inverse[0]}')
