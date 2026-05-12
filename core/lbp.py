import cv2
import numpy as np

def compute_lbp(image):
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image[i, j]
            binary = ''

            neighbors = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                image[i, j+1], image[i+1, j+1], image[i+1, j],
                image[i+1, j-1], image[i, j-1]
            ]

            for n in neighbors:
                binary += '1' if n >= center else '0'

            lbp[i, j] = int(binary, 2)

    return lbp


def compute_lbph(lbp, grid_x=8, grid_y=8):
    h, w = lbp.shape
    grid_h = h // grid_y
    grid_w = w // grid_x

    features = []

    for y in range(grid_y):
        for x in range(grid_x):
            grid = lbp[y*grid_h:(y+1)*grid_h,
                       x*grid_w:(x+1)*grid_w]

            hist = cv2.calcHist([grid], [0], None, [256], [0,256])
            hist = cv2.normalize(hist, hist).flatten()

            features.extend(hist)

    return np.array(features)