from typing import Union
import numpy as np

from fastapi import FastAPI
from utils.run_model import run

app = FastAPI()


@app.get("/generate")
def generate():
    input_map = np.random.randint(0, 7, size=(16, 16)) 

    x = run(0.2, 5, input_map)
    return {"generated_map": x}

@app.get("/")
def read_root():
    return {"Hello": "World"}

