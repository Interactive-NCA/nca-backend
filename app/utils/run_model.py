import pandas as pd
import numpy as np
from ribs.archives import ArchiveDataFrame
import torch as th
import json

from app.ext.pcgrl.control_pcgrl.evo.models import NCA, set_weights
from app.ext.pcgrl.control_pcgrl.evo.utils import get_one_hot_map

def run(symmetry, path_length, input_map):
    # Load init model parameters
    with open("app/models/model_init.json") as f:
        hyperameters = json.load(f)

    df = ArchiveDataFrame(pd.read_csv("app/models/trained_archive.csv"))

    distances = np.sqrt((df["behavior_0"] - symmetry)**2 + (df["behavior_1"] - path_length)**2)
    mask = distances == distances.min()
    start = list(df.columns).index("solution_0")
    weights = df[mask].iloc[:, start:].to_numpy().flatten() 


    model = NCA(**hyperameters)

    # Setup the model
    set_weights(model, weights=weights)

    N_TILE_TYPES = 8
    N_ITER = 50
    SET_FT = False

    # Get the level
    obs = get_one_hot_map(input_map, N_TILE_TYPES)
    in_tensor = th.unsqueeze(th.Tensor(obs), 0)

    for _ in range(N_ITER):
        if SET_FT:
            action, _ = model(in_tensor, fixed_tiles=[[7,1,1]].to_numpy())
        else:
            action, _ = model(in_tensor)
        
        level = th.argmax(action[0], dim=0)
        obs = get_one_hot_map(level.numpy(), N_TILE_TYPES)
        in_tensor = th.unsqueeze(th.Tensor(obs), 0)

    # Extract the level
    level = th.argmax(action[0], dim=0)

    return level.numpy().tolist()

def get_behaviour0_char():
    df = ArchiveDataFrame(pd.read_csv("app/models/trained_archive.csv"))
    bc1_min, bc1_max = df["behavior_0"].min(), df["behavior_0"].max()
    return bc1_min, bc1_max

def get_behaviour1_char():
    df = ArchiveDataFrame(pd.read_csv("app/models/trained_archive.csv"))
    bc2_min, bc2_max = df["behavior_1"].min(), df["behavior_1"].max()
    return bc2_min, bc2_max