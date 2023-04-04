import json
import pandas as pd
from ribs.archives import ArchiveDataFrame

def get_archive_path():
    """
    Get the path to the archive file
    """
    with open(f"app/models/model_path.json") as f:
        model_path = json.load(f)

    if model_path["EVAL"]:
        if model_path["FIXED"]:
            PATH = f'app/models/exp{model_path["EXP_ID"]}/fixed_tiles_evaluation/{model_path["METRIC"]}_eval_archive.csv'
        else:
            PATH = f'app/models/exp{model_path["EXP_ID"]}/evaluation/{model_path["METRIC"]}_eval_archive.csv'
    else:
        PATH = f'app/models/exp{model_path["EXP_ID"]}/trained_archive.csv'

    return PATH

def get_archive():
    """
    Get the archive
    """
    PATH = get_archive_path()

    raw_df = pd.read_csv(PATH)
    #sorted_df = raw_df.sort_values(by=['objective'], ascending=False)

    df = ArchiveDataFrame(raw_df)

    return df 

def get_model_settings():
    """
    Get the settings of the model
    """
    with open(f"app/models/model_path.json") as f:
        model_path = json.load(f)

    with open(f'app/models/exp{model_path["EXP_ID"]}/settings.json') as f:
        settings = json.load(f)
    
    return settings