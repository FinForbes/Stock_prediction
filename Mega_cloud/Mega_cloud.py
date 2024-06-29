from mega import Mega
from Stock_Models_Product_1.Cloud.utils import utils

if __name__ == "__main__":
    mega = Mega()
    m = mega.login("finforbesindia@gmail.com", "FinForbesIndia0911")

    # Define the local folder path to be uploaded
    local_folder_path = "../Cloud/script_data"

    # Upload the folder
    utils.upload_folder(m, local_folder_path, "script_data")
