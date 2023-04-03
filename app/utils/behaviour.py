from ribs.archives import ArchiveDataFrame
import pandas as pd

name = "app/models/playability_eval_archive.csv"

def get_extremes_behaviour0_char():
    """
    Get the extreme behaviour0 characteristics

    Returns:
        extreme0: List[float]
    """
    df = ArchiveDataFrame(pd.read_csv(name))
    bc1_min, bc1_max = df["behavior_0"].min(), df["behavior_0"].max()
    return bc1_min, bc1_max

def get_extremes_behaviour1_char():
    """
    Get the extreme behaviour1 characteristics

    Returns:
        extreme1: List[float]
    """
    df = ArchiveDataFrame(pd.read_csv(name))
    bc2_min, bc2_max = df["behavior_1"].min(), df["behavior_1"].max()
    return bc2_min, bc2_max

def get_behaviour0_char():
    """
    Get the behaviour0 characteristics

    Returns:
        behaviour0: List[float]
    """
    df = ArchiveDataFrame(pd.read_csv(name))
    return df["behavior_0"] 

def get_behaviour1_char():
    """
    Get the behaviour1 characteristics

    Returns:
        behaviour1: List[float]
    """
    df = ArchiveDataFrame(pd.read_csv(name))
    return df["behavior_1"] 

def get_obj_char():
    """
    Get the objective characteristics

    Returns:
        behaviours: List[List[float], List[float], List[float]]
    """
    df = ArchiveDataFrame(pd.read_csv(name))
    return df["behavior_0"].to_list(), df["behavior_1"].to_list(), df["objective"].to_list()