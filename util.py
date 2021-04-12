import bitstring as bs
import numpy as np

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