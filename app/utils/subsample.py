import pandas as pd
import pickle

def subsample_archive(archive_path, n_models):
    """
    Subsample the archive to the first n_models

    Args:
        archive_path (str): Path to the archive
        n_models (int): Number of models to subsample
    """
    # Load the archive
    df = pd.read_csv(archive_path)

    # Sort the DataFrame based on "objective" column in descending order
    df_sorted = df.sort_values("objective", ascending=False)

    # Select the first n_models rows of the sorted DataFrame
    df_top = df_sorted.head(n_models).copy()

    # Save the new DataFrame to a CSV file in a folder
    df_top.to_csv(archive_path, index=False)

def subsample_training_seeds(archive_path, training_seeds_path):
    """
    Subsample the training seeds to match the subsampled archive 

    Args:
      archive_path
      training_seeds_path
    """
    # Load the training seeds
    with open(training_seeds_path, "rb") as f:
        train_seeds = pickle.load(f)

    # Subsample the training seeds
    print("INSIDE")
    df = pd.read_csv(archive_path)

    generations = df["metadata"].unique()

    subsampled_seeds = {k: v for k, v in train_seeds.items() if k in generations}

    # Save the new training seeds to a pickle file
    print(df["metadata"])
    print(subsampled_seeds.keys())

    with open(training_seeds_path, "wb") as f:
        pickle.dump(subsampled_seeds, f)



if __name__ == "__main__":
    # Subsample the archive
    #subsample_archive("experiments/ExperimentId-6/trained_archive.csv", 100)

    # Subsample the training seeds
    #subsample_training_seeds("experiments/ExperimentId-6/trained_archive.csv", "experiments/ExperimentId-6/training_seeds.pkl")

    subsample_training_seeds("~/Downloads/models_exp14_trained_archive.csv", "training_seeds.pkl")