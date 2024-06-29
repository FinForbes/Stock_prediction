import os.path

from huggingface_hub import login, create_repo, upload_file, HfApi


def upload_model(local_dir, repo_id, hf_token):

    if not os.path.isdir(local_dir):
        raise ValueError(f"The directory{local_dir} not found")

    for root, _, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            remot_path = file

            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=repo_id,
                token=hf_token,
            )
