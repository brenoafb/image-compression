import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return (np.square(a - b)).mean(axis=None)