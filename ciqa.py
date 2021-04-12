import numpy as np
import PIL as pil
import matplotlib.pyplot as plt

def compress(image: np.ndarray, block_size: int, levels: int) -> dict:
    array = np.asarray(image)
    (n_blocks, blocks) = get_blocks(array, block_size)
    quantized_blocks = [quantize(block, levels) for block in blocks]
    return {
        'blocks_per_row': n_blocks,
        'block_size': block_size,
        'levels': levels,
        'quantized_blocks': quantized_blocks
    }

def decompress(data: dict) -> np.ndarray:
    n_blocks = data['blocks_per_row']
    block_size = data['block_size']
    quantized_blocks = data['quantized_blocks']
    size = block_size * n_blocks
    img = np.zeros((size,size), dtype=np.uint8)
    for k, (qblock, ranges) in enumerate(quantized_blocks):
        block_img = quantized_to_image(qblock, ranges)
        x = (k // n_blocks) * block_size
        y = (k % n_blocks) * block_size
        for i in range(block_size):
            for j in range(block_size):
                img[x+i,y+j] = block_img[i,j]
    return img

def get_blocks(mat: np.ndarray, block_size: int) -> tuple:
    dim = mat.shape[0] # assume square matrix
    n_blocks = dim // block_size  # number of blocks per row
    blocks = []
    for i in range(n_blocks):
        start1 = i * block_size
        end1 = (i + 1) * block_size
        for j in range(n_blocks):
            start2 = j * block_size
            end2 = (j + 1) * block_size
            block = mat[start1:end1, start2:end2]
            blocks.append(block)
    return (n_blocks, blocks)

def quantize(block: np.ndarray, levels: int) -> tuple:
    ranges = get_ranges(block, levels)
    quantized = np.zeros_like(block)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            quantized[i,j] = find_range(block[i,j], ranges)
    return (quantized, ranges)

def get_ranges(block: np.ndarray, levels: int) -> list:
    lo = block.min()
    hi = block.max()
    range_size = (hi - lo) / levels
    ranges = [
        (lo + i * range_size, lo + (i+1) * range_size) for i in range(levels)
    ]
    return ranges

def find_range(value: np.uint8, ranges: list) -> int:
    if value == ranges[-1][1]:
        return len(ranges) - 1
    for (i, r) in enumerate(ranges):
        if value >= r[0] and value < r[1]:
            return i
    print(f'Warning: returning none for value {value}, ({ranges})')
    return None

def quantized_to_image(quantized: np.ndarray, ranges: list) -> np.ndarray:
    img = np.zeros_like(quantized)
    midpoints = [(r[0] + r[1]) // 2 for r in ranges]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = midpoints[quantized[i,j]]
    return img
