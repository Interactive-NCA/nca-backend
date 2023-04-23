import numpy as np
from app.ext.pcgnca.pcgnca.evo._models import NCA, set_weights
from app.ext.pcgnca.pcgnca.evo._simulate import _simulate
from app.ext.pcgnca.pcgnca.evo._evaluate import ZeldaEvaluation
from app.utils.get_path import get_archive, get_model_settings


def run(exp_id, symmetry, path_length, input_map):
    """
    Find the closest model to the given symmetry and path length and run it on the input map

    Args:
        symmetry: float
        path_length: float
        input_map: List[List[List[int]]]

    Returns:
        generated_map: List[List[List[int]]]
    """

    settings = get_model_settings(exp_id)
    df = get_archive(exp_id)

    # For older archives that don't have metadata
    if "metadata" not in df.columns:
        df["metadata"] = 0

    # - Initialise evaluator for the experiment
    obj_weights = {
            "playability" : settings["playability_weight"],
            "reliability": settings["reliability_weight"]
        }
    evaluator = ZeldaEvaluation(settings["grid_dim"], obj_weights, settings["n_tiles"], settings["bcs"], settings["include_diversity"])

    # - Find the closest model
    distances = np.sqrt((df["measure_0"] - symmetry)**2 + (df["measure_1"] - path_length)**2)
    mask = distances == distances.min()
    start = list(df.columns).index("solution_0")
    weights = df[mask].iloc[:, start:-1].to_numpy().flatten() 
    generation = df[mask]["metadata"].to_numpy()[0]

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

    # - Run assertions
    # -- Model was not using bin. channel
    if not settings["binary_channel"]:
        bin_channel = None

    # - Run the model
    lvl_per_step, aux_channels, last_stats, batch_stats = _simulate(
        model,
        init_state,
        fixed_tiles,
        bin_channel,
        settings["n_tiles"],
        settings["n_steps"],
        settings["overwrite"],
        settings["padding_type"],
        obj_weights,
        "generated_lvls",
        settings["bcs"],
        settings["include_diversity"]
    )

    # - For dev purposes only
    # TODO: maybe add flag?
    print("="*100)
    # -- Input stats report
    input_stats = evaluator.get_zelda_level_stats(init_state[0])
    print(f"Input symmetry was: {input_stats['symmetry']}")

    # -- Output stats report
    batch_stats = [round(n, 2) for n in batch_stats]
    print(f"Training score of the selected model: {df[mask]['objective'].iloc[0]}")
    print(f"Path length: {last_stats['path_length']} (exp: {path_length})", f"Symmetry: {last_stats['symmetry']} (exp: {symmetry})")
    print(f"Objective: {batch_stats[0]}", f"Playbility penalty: {batch_stats[1]}")
    print("="*100)
    
    # - Get the expected shape, i.e., (n_aux_channels, n_steps, dim, dim)
    lvl_per_step = lvl_per_step.reshape((settings["n_steps"], dim, dim))
    aux_channels = np.transpose(aux_channels, (2, 0, 3, 4, 1))
    aux_channels = np.squeeze(aux_channels, axis=4)

    return [lvl_per_step.tolist(), aux_channels.tolist()] 