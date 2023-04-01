import pandas as pd
import numpy as np
from ribs.archives import ArchiveDataFrame
import torch as th

from app.ext.control_pcgnca.pcgnca.evo._models import NCA, set_weights
from app.ext.control_pcgnca.pcgnca.evo._simulate import _preprocess_input

def run(symmetry, path_length, input_map):
    """
    Find the closest model to the given symmetry and path length and run it on the input map

    Args:
        symmetry: float
        path_length: float
        input_map: List[List[int]]

    Returns:
        generated_map: List[List[int]]
    """

    # Hyperparameters
    N_TILE_TYPES = 5
    AUX_CHANNELS = 8
    N_ITER = 50
    SET_FT = False

    # Load the archive
    df = ArchiveDataFrame(pd.read_csv("app/models/trained_archive.csv"))

    # Find the closest model
    distances = np.sqrt((df["behavior_0"] - symmetry)**2 + (df["behavior_1"] - path_length)**2)
    mask = distances == distances.min()
    start = list(df.columns).index("solution_0")
    weights = df[mask].iloc[:, start:].to_numpy().flatten() 

    # Initialise the model
    model = NCA(N_TILE_TYPES, AUX_CHANNELS, SET_FT)

    # Set the weights
    set_weights(model, weights=weights)

    steps = []
    in_tensor =  _preprocess_input(input_map, N_TILE_TYPES, fixed=None, bin_mask=None, overwrite=False)

    for _ in range(N_ITER):
        # TODO: add fixed input
        action = model(in_tensor)
        level = th.argmax(action[0], dim=0)
        in_tensor =  _preprocess_input(level.numpy(), N_TILE_TYPES, fixed=None, bin_mask=None, overwrite=False)
        steps.append(level.numpy().tolist())

    return steps