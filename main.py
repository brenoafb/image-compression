import numpy as np
import PIL as pil
import matplotlib.pyplot as plt

def main():
    image = pil.Image.open('images/baboon.bmp')
    array = np.asarray(image)
    print(array)
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
