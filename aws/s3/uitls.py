import boto3
import os


class AWSUploader:
    def __init__(self, region_name):
        """
         # Use default credential provider chain, which will use the IAM role
        :param region_name: apt-south-1
        """

        self.s3_client = boto3.client("s3", region_name=region_name)
        self.ecr_client = boto3.client("ecr", region_name=region_name)

    def upload_file_to_s3(self, file_path, bucket_name, object_name=None):
        """

        :param file_path: The filewhich is needed to be uploaded to the s3 bucket
        :param bucket_name: The bucket for the storage (Please select accordingly)
        :param object_name: Type of file extension
        :return: None
        """
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name)
            print(
                f"File {file_path} uploaded successfully to {bucket_name}/{object_name}"
            )
            return True
        except Exception as e:
            print(f"Error uploading file to S3: {str(e)}")
            return False

    def upload_folder_to_s3(self, folder_path, bucket_name, prefix=""):
        """

        :param folder_path: Folder path relative the location in repository
        :param bucket_name: s3 Bucket name (Please select accordingly)
        :param prefix: Type of file extension
        :return: None
        """
        for root, dirs, files in os.walk(folder_path):
            for file in files:

                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, folder_path)
                s3_path = os.path.join(prefix, relative_path)

                self.upload_file_to_s3(local_path, bucket_name, s3_path)

class S3Uploader:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def upload_file_to_s3(self, file_path, bucket_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name)
            print(f"File {file_path} uploaded successfully to {bucket_name}/{object_name}")
            return True
        except Exception as e:
            print(f"Error uploading file to S3: {str(e)}")
            return False

def main():
    uploader = S3Uploader()
    bucket_name = "sagemaker-ap-south-1-533267146615"
    folder_path = "../../Stock_Models_Product_1/Cloud/script_data"  # Replace with the actual path to your CSV folder

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            uploader.upload_file_to_s3(file_path, bucket_name)

if __name__ == "__main__":
    main()