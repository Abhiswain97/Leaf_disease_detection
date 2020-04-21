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

# import model_tf

warnings.filterwarnings('ignore')


class Classify:
    def __init__(self):
        self.data = {}
        self.name = []
        self.energy = []
        self.ASM = []
        self.homogeneity = []
        self.dissimilarity = []
        self.contrast = []
        self.path = 'Labels\\RGB_superpixels'
        self.mask_label_path = 'Labels\\mask_ground_truth'
        # self.binary_labels = np.load(
        #     'Labels\\RGB_superpixels\\binary_labels(RGB).npy')
        # self.multiclass_labels = np.load(
        #     'Labels\\RGB_superpixels\\multiclass_labels(RGB).npy')

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
        for file, label in tqdm(zip(os.listdir(self.path), np.load('Labels\\RGB_superpixels\\binary_labels(RGB).npy'))):
            if file.endswith('binary.npy'):
                label_dict_binary[file] = label

        for file, label in tqdm(
                zip(os.listdir(self.path), np.load('Labels\\RGB_superpixels\\multiclass_labels(RGB).npy'))):
            if file.endswith('multiclass.npy'):
                label_dict_multiclass[file] = label

        with open('Labels\\mask_ground_truth\\binary_labels.json', 'w') as fp:
            json.dump(label_dict_binary, fp, default=convert)

        with open('Labels\\mask_ground_truth\\multiclass_labels.json', 'w') as fp:
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

        plt.savefig(f'results\\mask_prediction-{i + 1}.png')
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
        superpixel_names = sorted(glob.glob('Labels\\RGB_superpixels\\*_multiclass.npy'))
        masks = sorted(glob.glob('Labels\\mask_ground_truth\\*.npy'))

        labels = np.zeros((1, 2000))

        X = []
        with open('mask_prediction_features_and_labels.csv', mode='a+') as csv_file:
            csvwriter = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['R', 'G', 'B', 'Label'])

        for spxl_name, mask_name in tqdm(zip(superpixel_names, masks)):
            image = np.load(spxl_name)

            R = np.asarray(image[:, :, 0]).flatten()
            G = np.asarray(image[:, :, 1]).flatten()
            B = np.asarray(image[:, :, 2]).flatten()

            y = np.load(mask_name).flatten()

            for r, g, b, label in tqdm(zip(R, G, B, y)):
                with open('mask_prediction_features_and_labels.csv', mode='a+') as csv_file:
                    csvwriter = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow([r, g, b, label])

            print(f'{spxl_name} done !')
        # print('Starting Predictions')
        # for i in tqdm(range(20)):
        #     index = np.random.randint(2001, 2400)
        #
        #     image = np.load(superpixel_names[index])
        #
        #     r = image[:, :, 0]
        #     g = image[:, :, 1]
        #     b = image[:, :, 2]
        #
        #     for R, G, B in zip(r.flatten(), g.flatten(), b.flatten()):
        #         X.append([R, G, B])
        #
        #     y_true = np.load(masks[index])
        #     y_train = model.predict(X)
        #
        #     y_train = np.asarray(y_train)
        #
        #     self.visualize_predictions(y_true, y_train.reshape((image.shape[0], image.shape[1])), i)
        #
        #     X.clear()

    def NeuralNet(self):

        tfnet = model_tf.TfNet('features(binary_classify)(RGB).csv', 200, 32)
        model = tfnet.model()
        history = tfnet.train(model)
        # tfnet.plot(history)

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
    binary_file = 'features(binary_classify)(RGB).csv'
    multiclass_file = 'features(multiclass_classify)(RGB).csv'
    obj = Classify()
    is_binary = input('Is it binary classification? [True or False]: ')
    
    if is_binary == True:
        print('Binary classification!')
        print(f'Passing file: {binary_file} and is_binary={is_binary}')
        obj.classify(binary_file, is_binary)
    else:
        print('Multiclass classification!')
        print(f'Passing file: {multiclass_file} and is_binary={is_binary}')
        obj.classify(multiclass_file, is_binary)
