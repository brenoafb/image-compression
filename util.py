import bitstring as bs
import numpy as np
from scipy.cluster.vq import kmeans2

def get_blocks(mat: np.ndarray, block_size: int) -> tuple:
    '''
    Break a square matrix into square blocks of the specified size
    For best results, the size of the matrix should be a
    multiple of block_size
    '''
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

def get_codebook(vectors, k):
  '''
  given a collection of vectors and a number of clusters, 
  return a codebook and the label for each vector
  '''
  vectors = vectors.astype(np.float32)
  codebook, labels = kmeans2(vectors, k)
  codebook = codebook.astype(np.uint8)
  return (codebook, labels)

def bits2bytes(bits: bs.BitArray, shape: tuple, element_bit_length: int = 8) -> np.ndarray:
    values = []
    for i in range(0, bits.length, element_bit_length):
        curr_bits = bits[i:i+element_bit_length]
        values.append(curr_bits.uint)
    array = np.array(values).reshape(shape)
    return array

def bytes2bits(array: np.ndarray, element_bit_length: int = 8) -> bs.BitArray:
    shape = array.shape
    flattened = list(array.flatten())
    bitarrays = [bs.BitArray(uint=x, length=element_bit_length) for x in flattened]
    bits = bs.BitArray()
    for bitarray in bitarrays:
        bits.append(bitarray)
    return (bits, shape)