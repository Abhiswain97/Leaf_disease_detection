import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

from skimage.feature import greycomatrix, greycoprops
from skimage.segmentation import slic

from tqdm import tqdm
import json
import os
import sys
from dataclasses import dataclass
import csv
from collections import Counter
import glob

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.metrics import (
    sensitivity_specificity_support,
    geometric_mean_score,
    classification_report_imbalanced,
)
from imblearn.pipeline import Pipeline, make_pipeline

import models
import handle_imbalance


class Files:
    base_dir = sys.path[0][:sys.path[0].rindex('\\')+1]
    path = base_dir + "Labels\\RGB_superpixels"
    mask_label_path = base_dir + "Labels\\mask_ground_truth"
    binary_labels = np.load(
        base_dir + "Labels\\RGB_superpixels\\binary_labels(RGB).npy"
    )
    multiclass_labels = np.load(
        base_dir + "Labels\\RGB_superpixels\\multiclass_labels(RGB).npy"
    )


class FileUtils:
    @staticmethod
    def glcm_all_files(file_name):

        name, contrast, dissimilarity, homogeneity, corr, energy = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # with open(
        #     os.path.join(Files.base_dir, f"feature_files\\{file_name}.csv"),
        #     mode="w",
        #     newline="",
        # ) as metrics_csv:
        #     csv_writer = csv.writer(
        #                     metrics_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        #                 )
        #     csv_writer.writerow(
        #             ["name", "contrast", "dissimilarity", "homogeneity", "correlation", "energy"]
        #         )

        for file in tqdm(os.listdir(Files.path), total=len(os.listdir(Files.path))):
            if file.endswith("multiclass.npy"):
                image = np.load(os.path.join(Files.path, file))

                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                glcm = greycomatrix(image, [2], [0], 256, symmetric=True, normed=True)

                # Feature Extraction using GLCM
                name.append(file)
                contrast.append(greycoprops(glcm, "contrast")[0][0])
                dissimilarity.append(greycoprops(glcm, "dissimilarity")[0][0])
                homogeneity.append(greycoprops(glcm, "homogeneity")[0][0])
                corr.append(greycoprops(glcm, "correlation")[0][0])
                energy.append(greycoprops(glcm, "energy")[0][0])

                # name = file
                # contrast = greycoprops(glcm, "contrast")[0][0]
                # dissimilarity = greycoprops(glcm, "dissimilarity")[0][0]
                # homogeneity = greycoprops(glcm, "homogeneity")[0][0]
                # energy = greycoprops(glcm, "energy")[0][0]
                # corr = greycoprops(glcm, "correlation")[0][0]

                # with open(
                #     os.path.join(Files.base_dir, f"feature_files\\{file_name}.csv"),
                #     mode="a",
                #     newline="",
                # ) as metrics_csv:
                #     csv_writer = csv.writer(
                #         metrics_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                #     )
                #     csv_writer.writerow(
                #         [
                #             name,
                #             contrast,
                #             dissimilarity,
                #             homogeneity,
                #             corr,
                #             energy
                #         ]
                #     )

        data = {
            "name": name,
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "correlation": corr,
            "energy": energy,
            "label": Files.multiclass_labels,
        }

        features = pd.DataFrame(data)
        features.to_csv(
            Files.base_dir + f"\\feature_files\\{file_name}.csv", index=False
        )

    @staticmethod
    def make_json():
        def convert(o):
            if isinstance(o, np.int32):
                return int(o)
            raise TypeError

        label_dict_binary = {}
        label_dict_multiclass = {}
        for file, label in tqdm(zip(os.listdir(Files.path), Files.binary_labels)):
            if file.endswith("binary.npy"):
                label_dict_binary[file] = label

        for file, label in tqdm(zip(os.listdir(Files.path), Files.multiclass_labels)):
            if file.endswith("multiclass.npy"):
                label_dict_multiclass[file] = label

        with open(
            Files.base_dir + "Labels\\mask_ground_truth\\binary_labels.json", "w"
        ) as fp:
            json.dump(Files.base_dir + label_dict_binary, fp, default=convert)

        with open(
            Files.base_dir + "Labels\\mask_ground_truth\\multiclass_labels.json", "w"
        ) as fp:
            json.dump(label_dict_multiclass, fp, default=convert)


class FeatureUtils:
    @staticmethod
    def glcm(image):
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
            corr = greycoprops(glcm, "correlation")[0][0]

            features = np.array([contrast, dissimilarity, homogeneity, ASM, energy])

        return features


class PredictUtils:

    global model

    # Taken from: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    @staticmethod
    def make_confusion_matrix(
        cf,
        group_names=None,
        categories="auto",
        count=True,
        percent=True,
        cbar=True,
        xyticks=True,
        xyplotlabels=True,
        sum_stats=True,
        figsize=None,
        cmap="Blues",
        title=None,
        fig_name=None,
    ):
        """
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                    Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                    See http://matplotlib.org/examples/color/colormaps_reference.html
                    
        title:         Title for the heatmap. Default is None.
        """

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ["" for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = [
                "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
            ]
        else:
            group_percentages = blanks

        box_labels = [
            f"{v1}{v2}{v3}".strip()
            for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
        ]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score
                )
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize is None:
            # Get default figure size if not set
            figsize = plt.rcParams.get("figure.figsize")

        if not xyticks:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns_plot = sns.heatmap(
            cf,
            annot=box_labels,
            fmt="",
            cmap=cmap,
            cbar=cbar,
            xticklabels=categories,
            yticklabels=categories,
        )

        if xyplotlabels:
            plt.ylabel("True label")
            plt.xlabel("Predicted label" + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)

        if fig_name:
            figure = sns_plot.get_figure()
            figure.savefig(fig_name)

            print(f"Saved: {fig_name}")

    @staticmethod
    def get_metrics(y_true, y_pred, y_score=None, file_name=None):

        print(f"Accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}")

        print(classification_report_imbalanced(y_true=y_true, y_pred=y_pred))

        if y_score is not None:
            print(f"Roc-auc-score: {roc_auc_score(y_pred, y_score=y_score)}")
            plot_roc_curve(
                y_true=y_true, y_proba=y_score, n_classes=len(np.unique(y_true))
            )

        labels = ["True Neg", "False Pos", "False Neg", "True Pos"]

        if len(np.unique(y_true)) > 2:
            categories = ["Zero", "One", "Two"]
        else:
            categories = ["Zero", "One"]

        cf_matrix = confusion_matrix(y_true, y_pred)
        make_confusion_matrix(
            cf_matrix,
            group_names=labels,
            categories=categories,
            cmap="Blues",
            fig_name=file_name,
        )

    @staticmethod
    def pipeline_prediction(X, y, model, features, sampler="smotetomek"):
        
        sampler = handle_imbalance.Imbalance()(sampler=sampler)
        
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("sampler", sampler), ("model", model)]
        )

        pipe.fit(X, y)

        y_pred = pipe.predict(features)

        # PredictUtils.get_metrics(y_true=y, y_pred=y_pred, y_score=None)

        # return pipeline for direct prediction, skipping the details
        # receive the piepline and can call `get_metrics()` normally
        return y_pred

    # Taken from: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
    @staticmethod
    def plot_roc_curve(y_test, y_score, n_classes):

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label="ROC curve (area = %0.2f)" % roc_auc[i])
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right")
            plt.show()

    @staticmethod
    def predict_random_masks(model_name):

        superpixel_names = sorted(
            glob.glob(Files.base_dir + "Labels\\RGB_superpixels\\*_multiclass.npy")
        )
        masks = sorted(glob.glob(Files.base_dir + "Labels\\mask_ground_truth\\*.npy"))

        X = []
        for spxl_name, mask_name in tqdm(zip(superpixel_names, masks)):
            image = np.load(spxl_name)

            R = np.asarray(image[:, :, 0]).flatten()
            G = np.asarray(image[:, :, 1]).flatten()
            B = np.asarray(image[:, :, 2]).flatten()

            v = zip(R, G, B)
            for r, g, b in tqdm(v, total=len(R)):
                X.append([r, g, b])

            y = np.load(mask_name).flatten()

            print(np.array(X).shape, y.shape)

            print(f"{spxl_name} done !")

            model = models.ClassificationModels()(model_name, X, y)

            X.clear()
            break

        return model

        # print("Testing !")
        # idx = np.random.randint(2001, 2400)
        # image = np.load(superpixel_names[idx])
        #
        # R = np.asarray(image[:, :, 0]).flatten()
        # G = np.asarray(image[:, :, 1]).flatten()
        # B = np.asarray(image[:, :, 2]).flatten()
        #
        # v = zip(R, G, B)
        #
        # for r, g, b in tqdm(v, total=len(R)):
        #     X.append([r, g, b])
        #
        # y_true = np.load(masks[idx])
        # y_pred = model.predict(X)
        #
        # Visualize.visualize_predictions(
        #     y_true, np.array(y_pred).reshape(y_true.shape), idx
        # )
        #
        # print(np.array(X).shape, y.shape)
        #
        # X.clear()


class Visualize:
    @staticmethod
    def visualize_predictions(org_mask, predicted_mask, i):
        fig, axs = plt.subplots(1, 2)

        fig.suptitle("Original mask v/s Predicted mask")
        # fig.set_figheight(20)
        # fig.set_figwidth(20)

        axs[0].imshow(org_mask)
        axs[0].set_axis_off()
        axs[1].imshow(predicted_mask)
        axs[1].set_axis_off()

        plt.savefig(
            f"{Files.base_dir}results\\predicted_masks\\mask_prediction-{i + 1}.png"
        )
        plt.show()
