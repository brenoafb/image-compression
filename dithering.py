import numpy as np

def dither(img):
    # result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            old = img[i,j]
            new = closest_palette_color(old)
            img[i,j] = new
            qe = old - new
            if i < img.shape[0] - 1:
                img[i+1,j]   += qe * 7 // 16
            if i > 0 and j < img.shape[1] - 1:
                img[i-1,j+1] += qe * 3 // 16
            if j < img.shape[1] - 1:
                img[i  ,j+1] += qe * 5 // 16
            if i < img.shape[1] - 1 and j < img.shape[1] - 1:
                img[i+1,j+1] += qe * 1 // 16
            
    return img

# find closest web-safe color
# https://stackoverflow.com/a/29002582
def closest_palette_color(pixel):
    r = np.uint8(round( ( pixel[0] / 255.0 ) * 5 ) * 51)
    g = np.uint8(round( ( pixel[1] / 255.0 ) * 5 ) * 51)
    b = np.uint8(round( ( pixel[2] / 255.0 ) * 5 ) * 51)
    return np.array([r, g, b])