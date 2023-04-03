import pandas as pd
import numpy as np
from ribs.archives import ArchiveDataFrame
import json

from app.ext.pcgnca.pcgnca.evo._models import NCA, set_weights
from app.ext.pcgnca.pcgnca.evo._simulate import _simulate

def run(symmetry, path_length, input_map):
    """
    Find the closest model to the given symmetry and path length and run it on the input map

    Args:
        symmetry: float
        path_length: float
        input_map: List[List[List[int]]]

    Returns:
        generated_map: List[List[int]]
    """

    # Load the archive
    df = ArchiveDataFrame(pd.read_csv("app/models/playability_eval_archive.csv"))

    # Find the closest model
    distances = np.sqrt((df["behavior_0"] - symmetry)**2 + (df["behavior_1"] - path_length)**2)
    mask = distances == distances.min()
    start = list(df.columns).index("solution_0")
    weights = df[mask].iloc[:, start:].to_numpy().flatten() 

    # - Get settings of the experiment
    with open("app/models/settings.json") as f:
        settings = json.load(f)

    # - Get the model object
    model = NCA(settings["n_tiles"], settings["n_aux_chans"], settings["binary_channel"])

    # - Set the models' weights
    set_weights(model, weights=weights)

    # ------- Inference
    dim = input_map.shape[-1]
    input_map = input_map.reshape((2, 1, dim, dim))

    # - Extract init state
    init_state = input_map[0]

    # - Extract binary channel
    bin_channel = input_map[1]

    # - Infer the fixed tiles
    fixed_tiles = np.zeros((1, dim, dim), dtype=int)

    fixed_tiles[bin_channel == 1] = init_state[bin_channel == 1]

    # - Infer objective function weights from the settings
    obj_weights = {
            "playability" : settings["playability_weight"],
            "reliability": settings["reliability_weight"]
    }

    # - Run the model
    lvl_per_step = _simulate(
        model,
        init_state,
        fixed_tiles,
        bin_channel,
        settings["n_tiles"],
        settings["n_steps"],
        settings["overwrite"],
        obj_weights,
        "generated_lvls"
    )
    lvl_per_step = lvl_per_step.reshape((settings["n_steps"], dim, dim))
    return lvl_per_step.tolist()