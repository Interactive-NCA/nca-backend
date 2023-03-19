# 🎮 Neural Cellular Automata Zelda Level Generator 🧩

![Last commit](https://img.shields.io/github/last-commit/Interactive-NCA/nca-backend)

This repository contains a backend for hosting a neural cellular automata (NCA) model for generating Zelda levels. The model is built using PyTorch and is hosted using FastAPI and Google Cloud ☁️. 

The NCA model uses a grid-based approach, where each cell in the grid represents a tile in the level. The model generates levels by updating each cell in the grid based on its neighboring cells, and then repeating this process for a set number of iterations. The output of the model is a 2D grid of tiles that represents a generated Zelda level.

## ⚙️ Tech Stack
- FastAPI
- PyTorch 
- Google Cloud 

## 🚀 Getting Started

To get started with this project, you will need to clone the repository to your local machine:

```bash
git clone git@github.com:Interactive-NCA/nca-backend.git
```

Next, you will need to install the dependencies using pip:

```bash
pip install -r requirements.txt
```

Once you have installed the dependencies, you can run the backend using the following command:

```bash
uvicorn main:app --reload
```

This will start the FastAPI server, which you can access by navigating to `http://localhost:8000` in your web browser.

## 📝 Usage

The backend provides a simple API for generating Zelda levels using the NCA model. To generate a level, you can send a POST request to the `/generate` endpoint with the following JSON payload:

```json
{
}
```

The backend will respond with a JSON payload containing the generated level data:

```json
{
  "width": 32,
  "height": 32,
  "tiles": [
    [0, 0, 0, ...],
    [0, 1, 1, ...],
    [0, 1, 1, ...],
    ...
  ]
}

```

The width and height parameters specify the dimensions of the generated level in tiles. The tiles parameter is a 2D array of integers, where each integer represents a tile in the level.

## 🌐 Deployment

This repository is configured for deployment using Google Cloud️ ☁️. The application is re-deployed as a container on every push to the `main` branch.

## 🤝 Contributing

Contributions are always welcome! If you have any ideas or suggestions for the project, please create an issue or submit a pull request. Please follow these [conventions](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) for commit messages.

