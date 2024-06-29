import os


def rename_csv_files(directory) -> str:
    """
    Rename all CSV files in the specified directory by keeping only the prefix before the first underscore.

    :param directory: The path to the directory containing the CSV files
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        # if filename.endswith('.csv'):
        # Split the filename to extract the base name

        # Create the new filename
        new_filename = filename + ".csv"
        # Get the full path to the current file
        old_file = os.path.join(directory, filename)
        # Get the full path to the new file
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} to {new_filename}")

    print("All files have been renamed successfully.")


# Function to upload a folder recursively
def upload_folder(m, local_folder_path, remote_folder_path=""):
    # Create a new folder in Mega
    folder_name = remote_folder_path
    m.create_folder(folder_name)

    # Define the folder containing CSV files
    local_folder_path = local_folder_path

    # Loop through all files in the local folder
    for filename in os.listdir(local_folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(local_folder_path, filename)
            folder = m.find(folder_name)
            print(f"Uploading {filename} to {folder_name}...")
            m.upload(file_path, folder[0])
            print(f"{filename} uploaded successfully!")

    print("All files uploaded successfully!")
