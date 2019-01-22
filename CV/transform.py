import cv2, numpy as np
import matplotlib.pylab as plt
import matplotlib
plt.ion()


img = cv2.imread('/home/sanjay/Downloads/ferrari.jpg', 0)
rows, cols = img.shape

#translate
M = np.float32([[1, 0, 25], [0, 1, 25]])
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('translate.jpg', dst)

#rotate
M = cv2.getRotationMatrix2D((cols/2,rows/2), 30, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('rotate.jpg', dst)

#scale

