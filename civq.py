import numpy as np
from scipy.cluster.vq import kmeans, kmeans2, whiten
from util import *

def compress(image: np.ndarray, block_size: int, k: int) -> dict:
  size = image.shape[0] # assuming that image is square
  (_, blocks) = get_blocks(image, block_size)
  (codebook, labels) = get_codebook(blocks, k)
  return {
    'codebook': codebook,
    'labels': labels,
    'block_size': block_size,
    'size': size
  }

def decompress(data: dict) -> np.ndarray:
  codebook = data['codebook']
  labels = data['labels']
  size = data['size']
  block_size = data['block_size']

  return build_image_from_codebook(codebook, labels, size, block_size)

def build_image_from_codebook(codebook, labels, size, block_size):
  n_blocks = int((size / block_size) ** 2)
  img = np.zeros((size, size), dtype=np.uint8)
  blocks_per_row = size // block_size
  for p in range(n_blocks):
      block = codebook[labels[p]].reshape((block_size, block_size))
      x = (p // blocks_per_row) * block_size
      y = (p % blocks_per_row) * block_size
      for i in range(block_size):
          for j in range(block_size):
              img[x+i, y+j] = block[i, j]
  return img

def get_codebook(blocks, k):
  '''
  given the blocks of the image and a number of clusters, 
  return a codebook and the label for each block
  '''
  vectors = np.array([x.flatten() for x in blocks])
  vectors = vectors.astype(np.float32)
  codebook, labels = kmeans2(vectors, k)
  codebook = codebook.astype(np.uint8)
  return (codebook, labels)
