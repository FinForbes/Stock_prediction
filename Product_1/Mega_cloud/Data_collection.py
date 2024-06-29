from abc import ABC, abstractmethod
from mega import Mega
import yfinance as yf
import datetime as dt
import pandas as pd
import os


class CloudStorage(ABC):
    @abstractmethod
    def upload_file(self, location):
        pass

    @abstractmethod
    def download_file(self, remote_path, local_path):
        pass


class MegaCloud(CloudStorage):
    def __init__(self, email, password):
        self.mega = Mega()
        self.m = self.mega.login(email, password)

    def upload_file(self, location):
        self.m.upload(location)

    def download_file(self, remote_path, local_path):
        file = self.m.find(remote_path)
        self.m.download(file, dest_filename=local_path)


def data_collection(name):
    '''
    Collect the data from yfinance library and create a dataframe in the format
    where it collects High, Low, Open, Close and saves the dataframe.

    :param name: Company name (ticker symbol)
    :return: dataframe
    '''
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=5 * 365)

    # Fetching the data
    df = yf.download(name, start=start_date, end=end_date)

    # Selecting relevant columns
    df = df[['Open', 'High', 'Low', 'Close']]

    # Saving the dataframe as a CSV file
    file_name = f"{name}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
    df.to_csv(file_name)

    return df, file_name


def data_update(name, current_date, cloud_storage: CloudStorage):
    '''
    Provides an update to the data if it is older than one month.
    Downloads the file from Mega Cloud, updates it, and re-uploads the updated file.

    :param name: Name of the company listed under the ticker symbol
    :param current_date: Current date (YYYY-MM-DD)
    :param cloud_storage: Instance of a Cloud storage class implementing CloudStorage
    :return: updated dataframe
    '''
    current_date = dt.datetime.strptime(current_date, '%Y-%m-%d')
    one_month_ago = current_date - dt.timedelta(days=30)

    file_name = f"{name}_data.csv"

    # Download existing data from the Cloud
    local_path = file_name
    cloud_storage.download_file(file_name, local_path)

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)

        # Get the last date in the dataframe
        last_date = df.index[-1]

        if last_date < one_month_ago:
            new_data = yf.download(name, start=last_date + dt.timedelta(days=1), end=current_date)
            new_data = new_data[['Open', 'High', 'Low', 'Close']]

            # Append new data to the dataframe
            df = df.append(new_data)

            # Save updated dataframe
            df.to_csv(local_path)

            # Upload updated file to the Cloud
            cloud_storage.upload_file(local_path)

            # Delete the local file after uploading
            os.remove(local_path)
    else:
        # If file does not exist, create a new dataframe
        df, local_path = data_collection(name)
        cloud_storage.upload_file(local_path)

        # Delete the local file after uploading
        os.remove(local_path)

    return df


def upload_file_cloud(cloud_storage: CloudStorage, location):
    '''
    Uploads the file to the Cloud storage

    :param cloud_storage: Instance of a Cloud storage class implementing CloudStorage
    :param location: Path of the file to upload
    :return: None
    '''
    cloud_storage.upload_file(location)
    os.remove(location)  # Delete the file after uploading

def update_script_data(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Extract the stock ticker from the filename (assuming the filename format is 'TICKER.csv')
            ticker = os.path.splitext(filename)[0]
            file_ticker= filename.split('.')[0]
            # Create the full file path
            file_path = os.path.join(folder_path, filename)

            df_updated= yf.download(file_ticker)

            df= pd.read_csv(file_path)




