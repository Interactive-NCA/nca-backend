from typing import List
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.run_model import run
from app.utils.training_seeds import get_training_seeds
from app.utils.behaviour import get_extremes_behaviour0_char, get_extremes_behaviour1_char, get_behaviour0_char, get_behaviour1_char, get_obj_char
from app.utils.get_path import list_models_folders, get_model_settings


LOCAL = False

app = FastAPI(title='Interactive NCA API',
              description='API endpoints for the Interactive NCA model')

origins = [
    "http://localhost:3000",
    "https://localhost",
    "https://interactive-nca-ui-lukyrasocha.vercel.app/",
    "https://interactive-nca-ui.vercel.app/",
    "https://interactive-nca-ui-lukyrasocha.vercel.app",
    "https://interactive-nca-ui.vercel.app",
    "https://www.zeldalevelcraft.com/",
    "https://www.zeldalevelcraft.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hello world endpoint
@app.get("/")
def read_root():
    return {"Interactive": "NCA"}

# Generate a level from some given seed
@app.post("/generate")
async def generate(exp_id: int, path_length: float, symmetry: float, input_map: List[List[List[int]]]):
    """
    Generate a level from a given seed
    Args:
        exp_id: int
        path_length: float
        symmetry: float
        input_map: List[List[int]]

    Returns:
        generated_map: List[List[int]]
    """
    input_array = np.array(input_map[0]) # 2D int encoded level
    binary_array = np.array(input_map[1]) # binary array noting which tiles are fixed
    combined = np.array([input_array, binary_array])
    x = run(exp_id, symmetry, path_length, combined, LOCAL)
    return {"generated_map": x}


# Get behaviour characteristics
@app.get("/extreme-behaviour0")
def get_extreme_behaviour0():
    """
    Get the extreme behaviour0 characteristics

    Returns:
        extreme0: List[float]
    """
    return {"extreme0": get_extremes_behaviour0_char()}

@app.get("/extreme-behaviour1")
def get_extreme_behaviour1():
    """
    Get the extreme behaviour1 characteristics

    Returns:
        extreme1: List[float]
    """
    return {"extreme1": get_extremes_behaviour1_char()}

@app.get("/behaviour0")
def get_behaviour0():
    """
    Get the behaviour0 characteristics

    Returns:
        behaviour0: List[float]
    """
    return {"behaviour0": get_behaviour0_char()}

@app.get("/behaviour1")
def get_behaviour1():
    """
    Get the behaviour1 characteristics

    Returns:
        behaviour1: List[float]
    """
    return {"behaviour1": get_behaviour1_char()}

@app.get("/behaviours")
def get_both_and_obj(exp_id: int):
    """
    Get the behaviour0 and behaviour1 characteristics and the objective value

    Returns:
        behaviours: List[List[float], List[float], List[float]]
    """
    return {"behaviours": get_obj_char(exp_id, LOCAL)}

@app.get("/experimentnames")
def experiment_names():
    """
    Get experiment names

    Returns:
        experiments_names: List
    """
    return {"names": list_models_folders(LOCAL)}

@app.get("/trainingseeds")
async def get_training_seeds_endpoint(exp_id: int, path_length: float, symmetry: float):
    """
    Generate a level from a given seed
    Args:
        exp_id: int
        path_length: float
        symmetry: float

    Returns:
        training_seeds for selected model
    """

    x = get_training_seeds(exp_id, symmetry, path_length, LOCAL) 
    return {"training_seeds": x}

@app.get("/experimentdescriptions")
async def get_exp_desc(exp_id: int):
    """
    Return experiment's description

    Returns:
        desc: str   
    """
    settings = get_model_settings(exp_id, LOCAL)

    return {"desc": settings["settings_to_log"]["description"]}