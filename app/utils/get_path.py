import json
import pandas as pd
import os
from ribs.archives import ArchiveDataFrame
from google.cloud import storage

def get_archive_path():
    """
    Get the path to the archive file given in the model_path.json
    """
    with open(f"app/models/model_path.json") as f:
        model_path = json.load(f)

    if model_path["EVAL"]:
        if model_path["FIXED"]:
            PATH = f'app/models/exp{model_path["EXP_ID"]}/fixed_tiles_evaluation_summary/{model_path["METRIC"]}_eval_archive.csv'
        else:
            PATH = f'app/models/exp{model_path["EXP_ID"]}/evaluation_summary/{model_path["METRIC"]}_eval_archive.csv'
    else:
        PATH = f'models/exp{model_path["EXP_ID"]}/trained_archive.csv'

    return PATH

def get_archive(exp_id=None):
    """
    Get the archive
    """

    # - Load models
    PATH = get_archive_path()
    df = pd.read_csv(PATH)

    # - Select the best models with non-zero solution path length
    df = df.sort_values(by=['objective'], ascending=False)[df["measure_1"] != 0]
    df = ArchiveDataFrame(df)

    return df 

def get_model_settings(exp_id=None):
    """
    Get the settings of the model
    """
    if exp_id is None:
        with open(f"app/models/model_path.json") as f:
            model_path = json.load(f)

        with open(f'app/models/exp{model_path["EXP_ID"]}/settings.json') as f:
            settings = json.load(f)

        return settings
    else:
        BUCKET_NAME = "pcgnca-experiments"

        blob_path = f'models/exp{exp_id}/settings.json'
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        # Download the blob as a string
        blob_string = blob.download_as_string()

        # Load the JSON string as a dictionary
        settings = json.loads(blob_string)
    
    return settings

def list_models_folders():
    """List all folders in the 'models' folder of a Google Cloud Storage bucket."""
    bucket_name = "pcgnca-experiments"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    folders = set()
    
    for blob in bucket.list_blobs(prefix="models/"):
        if "/" in blob.name[len("models/"):]:
            folder = blob.name[len("models/"):].split("/")[0]
            folders.add(int(folder[3:]))
    
    return list(folders)