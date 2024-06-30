from huggingface_hub import login, create_repo, HfApi
from Product_1.Model.utils.huggingface_upload_models import upload_model

if __name__ == "__main__":
    # Your Hugging Face API token
    hf_token = "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"

    # Log in using your token
    login(token=hf_token, add_to_git_credential=True)

    # Define the repository details
    repo_id = "Finforbes/Stock_predictor"

    # Create the repository if it doesn't exist
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id)
        print(f"Repository '{repo_id}' already exists.")
    except:
        print(f"Creating repository '{repo_id}'...")
        create_repo(repo_id, token=hf_token, private=False)

    # Define the file path
    file_path = "../Cloud/models"

    upload_model(file_path, repo_id, hf_token)
    #
    # # Upload the file
    # upload_file(
    #     path_or_fileobj=file_path,
    #     path_in_repo='20MICRONS.h5',
    #     repo_id=repo_id,
    #     token=hf_token
    # )
