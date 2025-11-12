import cv2
import numpy as np
from helper import random_transform

img = cv2.imread('input.png')
bg = cv2.imread('background.png')
img = cv2.resize(img, (400, 400))

for i in range(100):
  warped, _ = random_transform(img, bg, 2.0)
  cv2.imshow('warped', warped)
  cv2.waitKey(0)

cv2.destroyAllWindows()


