import os
import pandas as pd
import pickle

def load_data(filepath):
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)


def save_data(df, filepath):
    """
    Save a Pandas DataFrame to a CSV file.

    Parameters:
    - df (DataFrame): Data to save.
    - filepath (str): Destination path.
    """
    df.to_csv(filepath, index=False)


def load_model(filepath):
    """
    Load a pickle model file.

    Parameters:
    - filepath (str): Path to the pickle file.

    Returns:
    - Model: Loaded model.
    """
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


def save_model(model, filepath):
    """
    Save a model to a pickle file.

    Parameters:
    - model (Model): Model to save.
    - filepath (str): Destination path.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def get_file_path(directory, filename):
    """
    Construct a filepath given a directory and filename.

    Parameters:
    - directory (str): Directory path.
    - filename (str): Filename.

    Returns:
    - str: Full filepath.
    """
    return os.path.join(directory, filename)
