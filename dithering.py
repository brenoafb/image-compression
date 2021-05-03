import numpy as np
from itertools import product

def dither(img):
    cp = np.copy(img).astype(np.float32) / 255.0
    for i in range(cp.shape[0]):       # top to bottom
        for j in range(cp.shape[1]):   # left to right
            old = cp[i,j]
            new = closest_palette_color(old)
            qe = old - new
            if j < cp.shape[1] - 1:
                cp[i,j+1] += qe * 7 / 16
            if i < cp.shape[0] - 1 and j > 0:
                cp[i+1,j-1] += qe * 3 / 16
            if i < cp.shape[0] - 1:
                cp[i+1,j] += qe * 5/16
            if i < cp.shape[0] - 1 and j < cp.shape[1] - 1:
                cp[i+1,j+1] += qe * 1 / 16
    return (cp * 255).astype(np.uint8)

def fit_to_palette(img):
    cp = np.copy(img).astype(np.float32) / 255.0
    for i in range(cp.shape[0]):
        for j in range(cp.shape[1]):
            new = closest_palette_color(cp[i,j])
            cp[i,j] = new
    return cp

def closest_palette_color(pixel):
    r = round(pixel[0])
    g = round(pixel[1])
    b = round(pixel[2])
    result = np.array([r, g, b], dtype=np.float32) 
    return result