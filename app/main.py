from typing import List
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.run_model import run, get_behaviour0_char, get_behaviour1_char

app = FastAPI()

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

@app.get("/test")
def generate():
    input_map = np.random.randint(0, 7, size=(16, 16)) 

    x = run(0.2, 5, input_map)
    return {"generated_map": x}

@app.get("/get-behaviour0")
def get_behaviour0():
    return {"behaviour0": get_behaviour0_char()}

@app.get("/get-behaviour1")
def get_behaviour1():
    return {"behaviour1": get_behaviour1_char()}

@app.get("/")
def read_root():
    return {"Interactive": "NCA"}

@app.post("/generate")
async def generate(path_length: float, symmetry: float, input_map: List[List[int]]):
    input_array = np.array(input_map)
    x = run(symmetry, path_length, input_array)
    return {"generated_map": x}
