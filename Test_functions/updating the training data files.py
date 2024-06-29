import os
import csv
import yfinance as yf
import pandas as pd
from datetime import datetime

# Specify the folder path containing your CSV files
folder_path = '../Cloud/script_data'

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Remove '.csv' and add '.NS' to the filename
        symbol = filename[:-4] + '.NS'

        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Read the existing CSV file
        try:
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            existing_data.index.name = 'Date'

            # Get the last date in the existing data
            last_date = existing_data.index[-1].strftime('%Y-%m-%d')

            # Fetch new data from Yahoo Finance
            new_data = yf.download(symbol, start=last_date, end=current_date)

            # Remove the last row of existing data to avoid duplication
            existing_data = existing_data.iloc[:-1]

            # Concatenate existing and new data
            updated_data = pd.concat([existing_data, new_data])

            # Remove any duplicate rows based on the index (Date)
            updated_data = updated_data[~updated_data.index.duplicated(keep='last')]

            # Sort the data by date
            updated_data.sort_index(inplace=True)

            # Save the updated data back to the CSV file
            updated_data.to_csv(file_path)

            print(f"Updated {filename} with new data from {last_date} to {current_date}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
