import warnings

warnings.filterwarnings("ignore")

import argparse
import csv
import json

# utility imports
import os
import sys
import glob
import joblib
from itertools import chain
from tqdm import tqdm
import operator

# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import slic
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from utils import PredictUtils, FeatureUtils, Visualize, Files
import slic_seg
import models
import handle_imbalance


class Classify:
    @staticmethod
    def fit(
        X_bin,
        y_bin,
        X_mul,
        y_mul,
        image,
        bin_model,
        mul_model,
        mask_predictor_name,
        sampler,
    ):

        # slic segment the image
        print("Segmenting image.....")
        slic_image = slic(image, n_segments=100)

        # separate the individual super-pixels
        print("Separating individual segments......")
        spxls = slic_seg.Superpixels.individual_superpixel(
            segments=slic_image, image=image, mode="RGB"
        )
        print(np.array(spxls).shape)
        # iterate over evert super-pixel
        features = FeatureUtils.glcm(spxls[0])

        print("Calculating GLCM features for each segment......")
        for spxl_bin in tqdm(spxls[1:]):
            features = np.vstack((features, FeatureUtils.glcm(spxl_bin)))

        print(features.shape)
        # Stage 1: binary classification

        print("Stage 1: binary classification")
        preds_bin = PredictUtils.pipeline_prediction(
            X=X_bin, y=y_bin, model=bin_model, sampler=sampler, features=features,
        )

        pred_pos_indices = np.where(preds_bin == 1)[0]

        print("Diseased super-pixel indices: ", pred_pos_indices)

        pos_spxls = operator.itemgetter(*pred_pos_indices)(spxls)

        # Stage 2: multi-class classification
        print("Stage 2: multiclass classification")
        features_mul = FeatureUtils.glcm(pos_spxls[0])

        for spxl_mul in tqdm(pos_spxls[1:]):
            features_mul = np.vstack((features_mul, FeatureUtils.glcm(spxl_mul)))

        preds_mul = PredictUtils.pipeline_prediction(
            X=X_mul, y=y_mul, model=mul_model, sampler=sampler, features=features_mul
        )

        pred_mul_indices = np.where(preds_mul >= 0)[0]

        mul_spxls = operator.itemgetter(*pred_mul_indices)(pos_spxls)

        print("PREDICTION: ")
        pred = (
            "Brown spot"
            if len(np.where(preds_mul == 1)[0]) > 0
            else "bacterial leaf blight"
        )

        print(pred)

        if mask_predictor_name is not None:
            Classify.mask_prediction(
                spxls=pos_spxls, model=mask_predictor_name, name=pred
            )

    @staticmethod
    def mask_prediction(spxls, model, name):

        for spxl in spxls:

            X = []

            R = np.asarray(spxl[:, :, 0]).flatten()
            G = np.asarray(spxl[:, :, 1]).flatten()
            B = np.asarray(spxl[:, :, 2]).flatten()

            v = zip(R, G, B)
            for r, g, b in tqdm(v, total=len(R)):
                X.append([r, g, b])

            preds = model.predict(X)

            plt.title(name)
            plt.imshow(preds.reshape(spxl.shape[:2]))
            plt.axis("off")
            plt.show()

            X.clear()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification parameters")

    parser.add_argument(
        "--binary_model",
        type=str,
        default="random_forest",
        choices=["svc", "xgboost", "random_forest", "logistic_regression", "knn"],
        help="Name of the model",
    )
    parser.add_argument(
        "--multiclass_model",
        type=str,
        default="random_forest",
        choices=["svc", "xgboost", "random_forest", "logistic_regression", "knn"],
        help="Name of the model",
    )
    parser.add_argument(
        "--mask_predict_model",
        type=str,
        default="random_forest",
        choices=["svc", "xgboost", "random_forest", "logistic_regression", "knn"],
        help="Predict mask model",
    )
    parser.add_argument("--sampler", type=str, help="sampler", default="smotetomek")
    parser.add_argument("--test_image_path", type=str, help="Path to test image")

    args = parser.parse_args()

    BASE_DIR = sys.path[0][:sys.path[0].rindex('\\')+1]
    FEATURE_DIR = "feature_files\\"
    binary_file = BASE_DIR + FEATURE_DIR + "new_glcm(binary).csv"
    multiclass_file = BASE_DIR + FEATURE_DIR + "new_glcm(multiclass).csv"

    obj = Classify()

    b_df = pd.read_csv(binary_file)
    m_df = pd.read_csv(multiclass_file)

    X_bin = b_df.loc[:, "contrast":"energy"].values
    y_bin = b_df["label"].values

    X_mul = b_df.loc[:, "contrast":"energy"].values
    y_mul = b_df["label"].values

    image = cv2.imread(args.test_image_path, 3)

    bin_model = models.ClassificationModels()(args.binary_model, X_bin, y_bin)
    mul_model = models.ClassificationModels()(args.multiclass_model, X_mul, y_mul)
    mask_model = PredictUtils.predict_random_masks(args.mask_predict_model)

    sampler = args.sampler

    obj.fit(
        X_bin=X_bin,
        y_bin=y_bin,
        X_mul=X_mul,
        y_mul=y_mul,
        image=image,
        bin_model=bin_model,
        mul_model=mul_model,
        mask_predictor_name=mask_model,
        sampler=sampler,
    )
