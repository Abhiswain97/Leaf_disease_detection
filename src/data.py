import os
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import slic

folder = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
seg_folder = ['Bacterial_leaf_blight_seg', 'Brown_spot_seg', 'leaf_smut_seg']
test_folder = 'Test'

path = r"C:\Users\Abhishek Swain\PycharmProjects\leaf-disease\Leaf disease"

'''image = cv2.imread(path + folder[0] + '\DSC_0365.JPG')
print(image)
cv2.imshow('hello', image)
cv2.waitKey(0)'''


def separate_clusters(label, size, image):
    for (i, segVal) in enumerate(np.unique(label)):
        # construct a mask for the segment
        print("[x] inspecting segment %d" % (i))
        mask = np.zeros(size[:2], dtype="uint8")
        mask[label == segVal] = 255

        cv2.imshow("Mask", mask)
        # cv2.resizeWindow("Mask", 100, 100)
        cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
        # cv2.resizeWindow("Mask", 100, 100)
        cv2.waitKey(100)


def segment(K, img):
    Z = img.reshape((-1, 3))

    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    separate_clusters(label, img.shape, res2)


def slic_segment(k, image):
    res = slic(image, 100)
    separate_clusters(res, image.shape, image)


for j, fname in enumerate(folder, 0):

    print(fname)
    file = os.listdir(path + '\\' + fname)
    start = time.time()

    for i in file:
        img = cv2.imread(path + '\\' + fname + '\\' + i)
        # segment(10, img)
        slic_segment(2, img)

    print(f'Time taken for {seg_folder[j]} is {time.time() - start} secs')
