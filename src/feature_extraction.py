from skimage.feature import greycomatrix, greycoprops
import os
import cv2
import pandas as pd
import time

path = "C:\\Users\\Abhishek Swain\\PycharmProjects\\leaf-disease\\Leaf disease"
seg_folder = ["Bacterial_leaf_blight_seg", "Brown_spot_seg", "leaf_smut_seg"]
folder = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

contrast = []
dissimilarity = []
homogeneity = []
ASM = []
energy = []
start = time.time()

for i in range(0, 3):
    print(seg_folder[i])
    fpath = os.listdir(path + "\\" + seg_folder[i])
    print(fpath)
    for fname in fpath:
        if "(KMeans=2)" not in fname:
            image = cv2.imread(path + "\\" + seg_folder[i] + "\\" + fname, 0)
            glcm = greycomatrix(
                image, distances=[1], angles=[0], normed=True, symmetric=True
            )

            # Feature Extraction using GLCM
            contrast.append(greycoprops(glcm, "contrast")[0][0])
            dissimilarity.append(greycoprops(glcm, "dissimilarity")[0][0])
            homogeneity.append(greycoprops(glcm, "homogeneity")[0][0])
            ASM.append(greycoprops(glcm, "ASM")[0][0])
            energy.append(greycoprops(glcm, "energy")[0][0])

            data = {
                "contrast": contrast,
                "dissimilarity": dissimilarity,
                "homogeneity": homogeneity,
                "ASM": ASM,
                "energy": energy,
            }

            features = pd.DataFrame(data)
            features.to_csv("features(disease).csv", index=False)

print(f"Completed in: {(time.time() - start) * 10} secs")
