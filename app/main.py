from typing import List
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.run_model import run
from app.utils.behaviour import get_extremes_behaviour0_char, get_extremes_behaviour1_char, get_behaviour0_char, get_behaviour1_char, get_obj_char

app = FastAPI(title='Interactive NCA API',
              description='API endpoints for the Interactive NCA model')

origins = [
    "http://localhost:3000",
    "https://localhost",
    "https://interactive-nca-ui-lukyrasocha.vercel.app/",
    "https://interactive-nca-ui.vercel.app/",
    "https://interactive-nca-ui-lukyrasocha.vercel.app",
    "https://interactive-nca-ui.vercel.app"
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
async def generate(path_length: float, symmetry: float, input_map: List[List[int]]):
    """
    Generate a level from a given seed
    Args:
        path_length: float
        symmetry: float
        input_map: List[List[int]]

    Returns:
        generated_map: List[List[int]]
    """
    input_array = np.array(input_map)
    x = run(symmetry, path_length, input_array)
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
def get_both_and_obj():
    """
    Get the behaviour0 and behaviour1 characteristics and the objective value

    Returns:
        behaviours: List[List[float], List[float], List[float]]
    """
    return {"behaviours": get_obj_char()}

# Test endpoints
@app.get("/test")
def generate():
    input_map = np.random.randint(0, 7, size=(16, 16)) 
    x = run(0.2, 5, input_map)
    return {"generated_map": x}