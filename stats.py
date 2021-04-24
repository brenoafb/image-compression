import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return (np.square(a - b)).mean(axis=None)

def psnr(mse, max=255):
    return 20 * np.log10(max) - 10 * np.log10(mse)