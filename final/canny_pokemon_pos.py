import numpy as np
import cv2
blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 10
min_line_length = 10
max_line_gap = 20


def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def canny_pokemon_pos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_vtx = np.array([[(320, img.shape[0]), (320, 0), (640, 0),
                        (640, img.shape[0])]])
    roi_edges = roi_mask(edges, roi_vtx)
    # cv2.imwrite('temp.jpg', roi_edges)
    k1 = roi_edges[0:119, :]
    k2 = roi_edges[90:209, :]
    k3 = roi_edges[180:299, :]
    k4 = roi_edges[270:389, :]
    k5 = roi_edges[360:479, :]
    k = [np.sum(k1), np.sum(k2), np.sum(k3), np.sum(k4), np.sum(k5)]
    return np.argmax(k)


if __name__ == '__main__':
    for i in range(20, 40):
        img = cv2.imread(f'E:\\documents\\hw\\Y3\\ROBOT\\THUEE_ROBOTS\\upstair_animal\\{i}.jpg')
        print(f'{i}.jpg:' + str(canny_pokemon_pos(img)))