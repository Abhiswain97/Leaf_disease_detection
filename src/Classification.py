import joblib
import os
import pickle
import json
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csv
import pandas as pd
import metrics
import numpy as np
from metrics import ClassificationMetrics
from handle_imbalance import Imbalance
from models import ClassificationModels

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import linear_model, tree, ensemble, metrics
from skopt import BayesSearchCV
import xgboost as xgb
from xgboost import XGBClassifier

from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')


class Classify:
    def __init__(self):
        self.base_dir = 'E:\\Btech project\\leaf-disease\\'
        self.data = {}
        self.name = []
        self.energy = []
        self.ASM = []
        self.homogeneity = []
        self.dissimilarity = []
        self.contrast = []
        self.path = self.base_dir + 'Labels\\RGB_superpixels'
        self.mask_label_path = self.base_dir + 'Labels\\mask_ground_truth'
        self.binary_labels = np.load(
            self.base_dir + 'Labels\\RGB_superpixels\\binary_labels(RGB).npy')
        self.multiclass_labels = np.load(
            self.base_dir + 'Labels\\RGB_superpixels\\multiclass_labels(RGB).npy')

    def plot_cm(self, y_true, y_pred, figsize=(10, 10)):
        cm = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true),
                          columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
        plt.show()

    def make_json(self):
        def convert(o):
            if isinstance(o, np.int32):
                return int(o)
            raise TypeError

        label_dict_binary = {}
        label_dict_multiclass = {}
        for file, label in tqdm(zip(os.listdir(self.path), self.binary_labels)):
            if file.endswith('binary.npy'):
                label_dict_binary[file] = label

        for file, label in tqdm(
                zip(os.listdir(self.path), self.multiclass_labels)):
            if file.endswith('multiclass.npy'):
                label_dict_multiclass[file] = label

        with open(self.base_dir + 'Labels\\mask_ground_truth\\binary_labels.json', 'w') as fp:
            json.dump(self.base_dir + label_dict_binary, fp, default=convert)

        with open(self.base_dir + 'Labels\\mask_ground_truth\\multiclass_labels.json', 'w') as fp:
            json.dump(label_dict_multiclass, fp, default=convert)

        # json.dump(label_dict_binary, 'Labels\\mask_ground_truth\\binary_labels.json')
        # json.dump(label_dict_multiclass, 'Labels\\mask_ground_truth\\multiclass_labels.json')

    def visualize_predictions(self, org_mask, predicted_mask, i):
        fig, axs = plt.subplots(1, 2)

        fig.suptitle('Original mask v/s Predicted mask', fontsize=50)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        axs[0].imshow(org_mask)
        axs[0].set_axis_off()
        axs[1].imshow(predicted_mask)
        axs[1].set_axis_off()

        # plt.savefig(f'results\\mask_prediction-{i + 1}.png')
        plt.show()

    def classify(self, file, is_binary):

        df = pd.read_csv(file)
        X = df[['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']]
        y = df['label']

        X_resampled, y_resampled = Imbalance()('repeated_edited_nearest_neighbours', X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.25)

        model = ClassificationModels()('svc', X_train, y_train, is_binary)

        print(ClassificationMetrics()('accuracy', y_test, model.predict(X_test)))

        if is_binary == 'True':
            results = {
                'model': model.__class__.__name__,
                'precision': ClassificationMetrics()('precision', y_test, model.predict(X_test)),
                'recall': ClassificationMetrics()('recall', y_test, model.predict(X_test)),
                'f1': ClassificationMetrics()('f1', y_test, model.predict(X_test)),
            }

            with open(f'results\\features(binary_classify)(RGB)-{model.__class__.__name__}.json', 'w') as file:
                json.dump(results, file)

            print(f'[SAVED] results\\features(binary_classify)(RGB)-{model.__class__.__name__}.json')

        else:
            results = {
                'model': model.__class__.__name__,
                'f1': ClassificationMetrics()('f1', y_test, model.predict(X_test)),
            }

            with open(f'results\\features(multiclass_classify)(RGB)-{model.__class__.__name__}.json', 'w') as file:
                json.dump(results, file)

            print(f'[SAVED] results\\features(multiclass_classify)(RGB)-{model.__class__.__name__}.json')

    def predict_mask(self):
        superpixel_names = sorted(glob.glob(self.base_dir + 'Labels\\RGB_superpixels\\*_multiclass.npy'))
        masks = sorted(glob.glob(self.base_dir + 'Labels\\mask_ground_truth\\*.npy'))

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

            print(f'{spxl_name} done !')

            model = ClassificationModels()(
                'random_forest', X, y, is_binary=True
            )

            X.clear()

            print('Testing !')
            idx = np.random.randint(2001, 2400)
            image = np.load(superpixel_names[idx])

            R = np.asarray(image[:, :, 0]).flatten()
            G = np.asarray(image[:, :, 1]).flatten()
            B = np.asarray(image[:, :, 2]).flatten()

            v = zip(R, G, B)

            for r, g, b in tqdm(v, total=len(R)):
                X.append([r, g, b])

            y_true = np.load(masks[idx])
            y_pred = model.predict(X)

            self.visualize_predictions(
                y_true, np.array(y_pred).reshape(y_true.shape), idx
            )

            print(np.array(X).shape, y.shape)

            X.clear()

    def glcm(self):
        for file in tqdm(os.listdir(self.path), total=len(os.listdir(self.path))):
            if file.endswith('binary.npy'):
                image = np.load(os.path.join(self.path, file))

                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                glcm = greycomatrix(image, distances=[1], angles=[
                    0], normed=True, symmetric=True)

                # Feature Extraction using GLCM
                self.name.append(file)
                self.contrast.append(greycoprops(glcm, 'contrast')[0][0])
                self.dissimilarity.append(
                    greycoprops(glcm, 'dissimilarity')[0][0])
                self.homogeneity.append(greycoprops(glcm, 'homogeneity')[0][0])
                self.ASM.append(greycoprops(glcm, 'ASM')[0][0])
                self.energy.append(greycoprops(glcm, 'energy')[0][0])

        self.data = {'name': self.name,
                     'contrast': self.contrast / np.mean(self.contrast),
                     'dissimilarity': self.dissimilarity,
                     'homogeneity': self.homogeneity,
                     'ASM': self.ASM,
                     'energy': self.energy,
                     'label': self.binary_labels
                     }

        features = pd.DataFrame(self.data)
        features.to_csv('features(binary_classify)(RGB).csv', index=False)


if __name__ == '__main__':
    FEATURE_DIR = 'feature_files\\'
    binary_file = FEATURE_DIR + 'features(binary_classify)(RGB).csv'
    multiclass_file = FEATURE_DIR + 'features(multiclass_classify)(RGB).csv'
    obj = Classify()

    obj.predict_mask()

    is_binary = input('Is it binary classification? [True or False]: ')

    if is_binary:
        print('Binary classification!')
        print(f'Passing file: {binary_file} and is_binary={is_binary}')
        obj.classify(binary_file, is_binary)
    else:
        print('Multiclass classification!')
        print(f'Passing file: {multiclass_file} and is_binary={is_binary}')
        obj.classify(multiclass_file, is_binary)
