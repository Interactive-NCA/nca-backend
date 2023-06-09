import numpy as np
from app.utils.get_path import get_archive, load_training_seeds

def get_training_seeds(exp_id, symmetry, path_length, local):
    """
    Find the model and its training seeds 

    Args:
        symmetry: float
        path_length: float
        local: bool

    Returns:
        List[List[int]]
    """

    # - Load the archive
    df = get_archive(exp_id, local)

    # - Find the closest model
    distances = np.sqrt((df["measure_0"] - symmetry)**2 + (df["measure_1"] - path_length)**2)
    mask = distances == distances.min()
    generation = df[mask]["metadata"].to_numpy()[0]


    # - Apply the fixed tiles
    training_seeds = load_training_seeds(exp_id, local)

    # Get the first key of the dictionary to check whether the dictionaries within the dictionary have fixed tiles, binary mask and init states 
    first_key = list(training_seeds.keys())[0]

    if len(training_seeds[first_key]) == 3:
        init_state, fixed_state, binary_mask = training_seeds[generation]["init_states"], training_seeds[generation]["fixed_states"], training_seeds[generation]["binary_mask"]
    else:
        init_state = training_seeds[generation]["init_states"]
        binary_mask = np.zeros_like(init_state)
        fixed_state = np.zeros_like(init_state)

    np.putmask(init_state, binary_mask, fixed_state)

    return [init_state.tolist(), binary_mask.tolist()]


