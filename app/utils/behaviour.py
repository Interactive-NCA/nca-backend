from app.utils.get_path import get_archive

def get_extremes_behaviour0_char():
    """
    Get the extreme behaviour0 characteristics

    Returns:
        extreme0: List[float]
    """
    df = get_archive()
    bc1_min, bc1_max = df["measure_0"].min(), df["measure_0"].max()
    return bc1_min, bc1_max

def get_extremes_behaviour1_char():
    """
    Get the extreme behaviour1 characteristics

    Returns:
        extreme1: List[float]
    """
    df = get_archive()
    bc2_min, bc2_max = df["measure_1"].min(), df["measure_1"].max()
    return bc2_min, bc2_max

def get_behaviour0_char():
    """
    Get the behaviour0 characteristics

    Returns:
        behaviour0: List[float]
    """
    df = get_archive()
    return df["measure_0"] 

def get_behaviour1_char():
    """
    Get the behaviour1 characteristics

    Returns:
        behaviour1: List[float]
    """
    df = get_archive()
    return df["measure_1"] 

def get_obj_char(exp_id: int, local=False):
    """
    Get the objective characteristics

    Returns:
        behaviours: List[List[float], List[float], List[float]]
    """
    df = get_archive(exp_id, local)
    return df["measure_0"].to_list(), df["measure_1"].to_list(), df["objective"].to_list()