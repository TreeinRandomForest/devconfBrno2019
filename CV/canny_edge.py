import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
plt.ion()

def read_file(filename):
    return cv2.imread(filename, 0)

def apply_filter(img, filter):
    img_rows, img_cols = img.shape
    filter_rows, filter_cols = filter.shape

    result = np.zeros((img_rows-filter_rows+1, img_cols-filter_cols+1))

    for i in np.arange(result.shape[0]):
        for j in np.arange(result.shape[1]):
            result[i,j] = np.sum(img[i:i+filter_rows, j:j+filter_cols] * filter)

    return result

def edge(img):
    vert = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    horz = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

    r_vert = apply_filter(img, vert)
    r_horz = apply_filter(img, horz)

    return r_vert, r_horz

if __name__=='__main__':
    img = read_file('road-trip.jpg')

    r1, r2 = edge(img)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig("road_trip_grayscale.jpg", transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.imshow(r1, cmap='gray')
    plt.axis('off')
    plt.savefig("road_trip_r1.jpg", transparent=True, bbox_inches='tight')
    plt.clf()

    plt.imshow(r2, cmap='gray')
    plt.axis('off')
    plt.savefig("road_trip_r2.jpg", transparent=True, bbox_inches='tight')
    plt.clf()

    plt.imshow(cv2.Canny(img, 220, 240), cmap='gray') 
    plt.axis('off')
    plt.savefig("Canny.jpg", transparent=True, bbox_inches='tight')
    plt.clf()