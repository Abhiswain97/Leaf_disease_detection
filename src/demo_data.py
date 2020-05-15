import os
from typing import List

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import greycomatrix, greycoprops
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)

seg_folder: List[str] = ['Bacterial_leaf_blight_seg', 'Brown_spot_seg', 'leaf_smut_seg']


class Data(object):

    def __init__(self, path):
        """
        Initialization
        :param path:
        """
        self.path = path
        self.contrast = []
        self.dissimilarity = []
        self.homogeneity = []
        self.ASM = []
        self.energy = []
        self.data = {}

    def rgb_to_hsv(self, image):
        """
        Takes an RGB Image and converts it into HSV Image
        :param image:
        :return HSV Image:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def seperate_cluster(self, image: object, label: object) -> object:
        """
        Takes an Image and separates each individual clusters
        :param image:
        :param label:
        """
        # loop over the unique segment values
        for i, segVal in enumerate(np.unique(label)):
            # print("[x] inspecting segment %d" % (i))

            # Create a mask of the size of the image and fill tit with zeros
            mask = np.zeros(image.shape[:2], dtype="uint8")
            
            # Check for the labels for segments in the labels array and make the corresponding label a value of 255.
            # It was found that 4 clusters is the ideal number and the disease region is present in the 4th cluster.
            mask[label.reshape(image.shape[:2]) == 3] = 255

            # Perform a bitwise and to regain the color aspect of the mask
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            # cv2.imshow('Leaf Segment', segmented_image)
            # cv2.waitKey(1000)
            return segmented_image

    # TODO: Write a function for glcm feature extraction

    def glcm(self):
        for i in range(0, 3):
            print(seg_folder[i])
            fpath = os.listdir(self.path + '\\' + seg_folder[i])
            print(fpath)
            for fname in fpath:
                if True:
                    image = cv2.imread(self.path + '\\' + seg_folder[i] + '\\' + fname)
                    Hue = image[:, :, 0]
                    Sat = image[:, :, 1]

                    glcm = greycomatrix(image, distances=[1], angles=[0], normed=True, symmetric=True)

                    # Feature Extraction using GLCM
                    self.contrast.append(greycoprops(glcm, 'contrast')[0][0])
                    self.dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0][0])
                    self.homogeneity.append(greycoprops(glcm, 'homogeneity')[0][0])
                    self.ASM.append(greycoprops(glcm, 'ASM')[0][0])
                    self.energy.append(greycoprops(glcm, 'energy')[0][0])

                    self.data = {'contrast': self.contrast,
                                 'dissimilarity': self.dissimilarity,
                                 'homogeneity': self.homogeneity,
                                 'ASM': self.ASM,
                                 'energy': self.energy,
                                 }

        features = pd.DataFrame(self.data)
        features.to_csv('features(disease).csv', index=False)

    # TODO: Try different classification techniques

    def recreate_image(self, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        cv2.imshow('image', image)
        cv2.waitKey(0)
        exit(0)
        return image

    def segment_and_store(self):
        """
        Segments the image and stores it in corresponding directory
        """
        for i, folder in enumerate(os.listdir(self.path)):
            print(folder)
            for file in tqdm(os.listdir(os.path.join(self.path, folder))):

                if os.path.exists(
                        os.path.join(self.path, seg_folder[i] + f'\\(disease){file}')) or folder.endswith('seg'):
                    logging.info("[Segmented Image Already Exists] in " + f'{folder}')
                else:
                    print(os.path.join(self.path, folder + f'\\{file}'))
                    image = cv2.imread(os.path.join(self.path, folder + f'\\{file}')).astype('uint8')
                    image = cv2.resize(image, (512, 512))
                    # image = image.reshape((-1, 3))

                    hsv_image = self.rgb_to_hsv(image).reshape((-1, 3))  # TODO: Handle image conversion to hsv errors
                    kmeans = KMeans(n_clusters=4, random_state=0).fit(hsv_image)  # TODO: KMeans Elbow method and #
                    # Dendrogram
                    w, h, d = tuple(image.shape)
                    self.recreate_image(kmeans.cluster_centers_, kmeans.labels_, w, h)
                    print(kmeans.labels_)
                    segmented_image = self.seperate_cluster(hsv_image.reshape(image.shape), kmeans.labels_)
                    # cv2.imwrite(os.path.join(self.path, seg_folder[i] + f'\\(disease){file}'), segmented_image)
                    logging.info('[SAVED] {0}'.format(os.path.join(self.path, seg_folder[i] + f"\\(disease){file}")))


if __name__ == '__main__':
    image_path = r'C:\Users\Abhishek Swain\PycharmProjects\leaf-disease\Leaf disease'
    data = Data(image_path)
    data.segment_and_store()
    # data.glcm()
