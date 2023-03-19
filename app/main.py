from typing import Union
import numpy as np

from fastapi import FastAPI
from app.utils.run_model import run

app = FastAPI()

@app.get("/generate")
def generate():
    input_map = np.random.randint(0, 7, size=(16, 16)) 

    x = run(0.2, 5, input_map)
    return {"generated_map": x}

@app.get("/")
def read_root():
    return {"Hello": "World"}

#@app.post("/generate2")
#def generate(symmetry: float, path_length: int, input_map: Union[list, np.ndarray]):
#    x = run(symmetry, path_length, input_map)
#    return {"generated_map": x}
