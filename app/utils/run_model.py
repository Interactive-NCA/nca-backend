import numpy as np
from app.ext.pcgnca.pcgnca.evo._models import NCA, set_weights
from app.ext.pcgnca.pcgnca.evo._simulate import _simulate
from app.utils.get_path import get_archive, get_model_settings

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

    settings = get_model_settings()
    df = get_archive()

    # Find the closest model
    distances = np.sqrt((df["measure_0"] - symmetry)**2 + (df["measure_1"] - path_length)**2)
    mask = distances == distances.min()
    start = list(df.columns).index("solution_0")
    weights = df[mask].iloc[:, start:].to_numpy().flatten() 

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
    lvl_per_step, aux_channels, last_stats = _simulate(
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

    print("="*15)
    print("Last stats: ", last_stats)
    print("="*15)
    lvl_per_step = lvl_per_step.reshape((settings["n_steps"], dim, dim))

    # Get the shape (n_aux_channels, n_steps, dim, dim)
    aux_channels = np.transpose(aux_channels, (2, 0, 3, 4, 1))
    aux_channels = np.squeeze(aux_channels, axis=4)

    return [lvl_per_step.tolist(), aux_channels.tolist()]