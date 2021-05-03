import numpy as np
import json
from base64 import b64encode, b64decode
from util import *
from math import ceil, log2

def compress(image: np.ndarray, k: int) -> dict:
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

def serialize(data):
    codebook_size = len(data['codebook'])
    nbits = ceil(log2(codebook_size))
    bits = []
    (codebook_bits, codebook_shape) = bytes2bits(data['codebook'])
    codebook_b64 = b64encode(codebook_bits.tobytes()).decode()
    (labels_bits, labels_shape) = bytes2bits(data['labels'], element_bit_length=nbits)
    labels_b64 = b64encode(labels_bits.tobytes()).decode()
    size = data['size']
    new_data = {'size': size,
                'csize': codebook_size,
                'cshape': codebook_shape,
                'cb64': codebook_b64,
                'lshape': labels_shape,
                'lb64': labels_b64
              }
    return json.dumps(new_data)

def deserialize(serialized):
    data = json.loads(serialized)
    nbits = ceil(log2(data['csize']))
    
    codebook_bits = bs.BitArray(b64decode(data['cb64']))
    codebook = bits2bytes(codebook_bits, shape=data['cshape'])
    labels_bits = bs.BitArray(b64decode(data['lb64']))
    labels = bits2bytes(labels_bits, shape=data['lshape'], element_bit_length=nbits)
    size = data['size']

    new_data = {
      'size': size,
      'codebook': codebook,
      'labels': labels
    }
    return new_data

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

  return img
