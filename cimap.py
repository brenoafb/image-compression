import numpy as np
from util import *

def compress(image: np.ndarray, block_size: int, k: int) -> dict:
  size = image.shape[0] # assuming that image is square
  vectors = get_vectors(image)
  (codebook, labels) = get_codebook(vectors, k)
  return {
    'codebook': codebook,
    'labels': labels,
    'size': size
  }

def decompress(data: dict) -> np.ndarray:
  codebook = data['codebook']
  labels = data['labels']
  size = data['size']

  return build_image_from_codebook(codebook, labels, size)

def get_vectors(array: np.ndarray) -> np.ndarray:
  r = array[:,:,0].flatten()
  g = array[:,:,1].flatten()
  b = array[:,:,2].flatten()
  return np.array([np.array(list(x)) for x in zip(r,g,b)])

def build_image_from_codebook(codebook, labels, size):
  img = np.zeros((size, size, 3), dtype=np.uint8)
  for i in range(size):
    for j in range(size):
      pixel = codebook[labels[i * size + j]]
      img[i][j] = pixel

  # for p in range(n_blocks):
  #     block = codebook[labels[p]].reshape((block_size, block_size))
  #     x = (p // blocks_per_row) * block_size
  #     y = (p % blocks_per_row) * block_size
  #     for i in range(block_size):
  #         for j in range(block_size):
  #             img[x+i, y+j] = block[i, j]
  return img
