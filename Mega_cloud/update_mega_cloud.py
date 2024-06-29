import os
from mega import Mega
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(filename='../Test_functions/update_csv.log', level=logging.INFO, format='%(asctime)s:%(message)s')

# Mega login credentials
Mega_login = {
    'email': 'finforbesindia@gmail.com',
    'password': 'FinForbesIndia0911'
}


def data_csv_from_mega(mega, file_name, download_path, folder_name='script_data'):
    """
    Downloads a CSV file from a specific folder in Mega to the specified download path.

    Args:
        mega (Mega): An instance of the Mega class.
        file_name (str): The name of the file to download.
        download_path (str): The path to download the file to.
        folder_name (str): The name of the folder in Mega containing the file.

    Returns:
        str: The path to the downloaded file, or None if an error occurred.
    """
    try:
        # Find the folder
        folder = mega.find(folder_name)
        if not folder:
            logging.error(f"Folder {folder_name} not found on Mega.")
            return None

        # Find the file within the folder
        file = mega.find(file_name, folder[0])
        if not file:
            logging.error(f"File {file_name} not found in folder {folder_name} on Mega.")
            return None

        # Ensure the download path exists
        os.makedirs(download_path, exist_ok=True)

        # Download the file to the specified path
        full_path = os.path.join(download_path, file_name)
        mega.download(file[0], dest_path=full_path)

        logging.info(f"Downloaded {file_name} from {folder_name} to {full_path}.")
        return full_path
    except Exception as e:
        logging.error(f"Error downloading file {file_name}: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {e.args}")
        return None


def download_with_retry(mega, file_name, download_path, max_retries=3, delay=5):
    for attempt in range(max_retries):
        result = data_csv_from_mega(mega, file_name, download_path)
        if result:
            return result
        logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
        time.sleep(delay)
    logging.error(f"Failed to download {file_name} after {max_retries} attempts.")
    return None


def main():
    download_path = './+'
    scripts_file = '../Cloud/Indian_scripts_name/filtered_file.csv'

    # Read the scripts file
    df = pd.read_csv(scripts_file)
    scripts = df['Yahoo_Equivalent_Code']

    # Login to Mega
    mega = Mega()
    m = mega.login(Mega_login['email'], Mega_login['password'])

    # Get list of CSV files from the 'training_data' folder in Mega
    folder = m.find('script_data')
    if not folder:
        logging.error("'training_data' folder not found in Mega.")
        return

    files = m.get_files_in_node(folder[0])
    file_names = [file['a']['n'] for file in files.values() if file['a']['n'].endswith('.csv')]

    file_processed = 0
    for file_name in file_names:
        print(f"Downloading: {file_name}")
        downloaded_file_path = download_with_retry(m, file_name, download_path)

        if downloaded_file_path:
            print(f"Successfully downloaded: {downloaded_file_path}")
            # Process the downloaded file (if needed)
            # df = pd.read_csv(downloaded_file_path)
            # Perform your data processing here
        else:
            print(f"Failed to download: {file_name}")

        file_processed += 1
        if file_processed >= 3:
            break


if __name__ == '__main__':
    main()