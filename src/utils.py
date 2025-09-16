import pandas as pd
import numpy as np
import joblib
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(model, filepath)

def load_model(filepath: str):
    """Load a trained model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)

def calculate_wcss(model, data: pd.DataFrame) -> float:
    """Calculate Within-Cluster-Sum-of-Squares (WCSS) for clustering."""
    if hasattr(model, "inertia_"):
        return model.inertia_
    else:
        raise AttributeError("Model does not have inertia_ attribute (not supported).")

def get_cluster_counts(labels: np.ndarray) -> dict:
    """Return cluster distribution as a dictionary."""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))



def log_message(message: str) -> None:
    """Simple logger for console output."""
    print(f"[INFO] {message}")
