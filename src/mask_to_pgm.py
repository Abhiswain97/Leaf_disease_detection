import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = r"Leaf disease"
mask_folders = ["Brown spot masks", "leaf smut masks"]

for file in os.listdir("Leaf_disease\\leaf smut masks"):
    print(file.split(".")[0])
    pgm_mask = cv2.imread(os.path.join("Leaf_disease\\leaf smut masks", file), -1)
    cv2.imshow("pgm", pgm_mask)
    cv2.waitKey(100)
