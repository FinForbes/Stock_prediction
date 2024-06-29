from Stock_Models_Product_1.Mega_cloud.Mega_cloud import data_collection, MegaCloud, upload_file_cloud


# Example usage:
# # Collect data and upload it to MegaCloud
# df, file_name = data_collection('AAPL')
# cloud_storage = MegaCloud()
# upload_file_cloud(cloud_storage, file_name)
#
# # # Update data and upload it to MegaCloud
# # updated_df = data_update('AAPL', dt.datetime.now().strftime('%Y-%m-%d'), cloud_storage)


import csv

# Define the file name or path to your CSV file
csv_file = "../Cloud/Indian_scripts_name/filtered_file.csv"

# Initialize lists to store the columns
# company_names = []
yahoo_codes = []

# Read the CSV file
with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    # Iterate over each row in the CSV
    for row in reader:
        # company_names.append(row['NAME OF COMPANY'])
        yahoo_codes.append(row["Yahoo_Equivalent_Code"])

# Output the lists
# print("Company Names:", company_names)
# print("Yahoo Codes:", yahoo_codes)

cloud_storage = MegaCloud("finforbesindia@gmail.com", "FinForbesIndia0911")
for symbol in yahoo_codes:
    # Collect data and get the file name
    df, file_name = data_collection(symbol)
    print(f"Data collected for {symbol}. Saving to {file_name}")

    # Upload the file to the Cloud
    upload_file_cloud(cloud_storage, file_name)
    print(f"File {file_name} uploaded to the Cloud.")
