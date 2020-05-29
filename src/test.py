from slic import individual_superpixel
import cv2
import os
from skimage.feature import greycomatrix, greycoprops


def glcm(self):
    image = cv2.imread("test.jpg")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        glcm = greycomatrix(
            image, distances=[1], angles=[0], normed=True, symmetric=True
        )

        contrast = greycoprops(glcm, "contrast")[0][0]
        dissimilarity = greycoprops(glcm, "dissimilarity")[0][0]
        homogeneity = greycoprops(glcm, "homogeneity")[0][0]
        ASM = greycoprops(glcm, "ASM")[0][0]
        energy = greycoprops(glcm, "energy")[0][0]

    features = [contrast, dissimilarity, homogeneity, ASM, energy]
