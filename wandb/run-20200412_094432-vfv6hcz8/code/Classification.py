import os
import pickle
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, ensemble, metrics
from xgboost import XGBClassifier

from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm
import functools
import time
import warnings
import model_tf

warnings.filterwarnings('ignore')

from joblib import Memory, Parallel, delayed
location = 'cachedir'
memory = Memory(location, verbose=0)


class Classify:
    def __init__(self, file):
        self.file = file
        self.data = {}
        self.name = []
        self.energy = []
        self.ASM = []
        self.homogeneity = []
        self.dissimilarity = []
        self.contrast = []
        self.path = 'Labels\\RGB_superpixels'
        self.mask_label_path = 'Labels\\mask_ground_truth'
        self.binary_labels = np.load('Labels\\RGB_superpixels\\binary_labels(RGB).npy')
        self.multiclass_labels = np.load('Labels\\RGB_superpixels\\multiclass_labels(RGB).npy')
        self.models = {
            'logistic': linear_model.LogisticRegression(),
            'decision_tree': tree.DecisionTreeClassifier(),
            'random_forest': ensemble.RandomForestClassifier()
        }

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

    def metrics(self, name, y_true, y_pred):
        if name == 'f1_score':
            if len(np.unique(y_pred)) > 2:
                return metrics.f1_score(y_true, y_pred, average='weighted')
            else:
                return metrics.f1_score(y_true, y_pred)
        if name == 'recall':
            if len(np.unique(y_pred)) > 2:
                return metrics.recall_score(y_true, y_pred, average='weighted')
            else:
                return metrics.recall_score(y_true, y_pred)
        if name == 'precision':
            if len(np.unique(y_pred)) > 2:
                return metrics.precision_score(y_true, y_pred, average='weighted')
            else:
                return metrics.precision_score(y_true, y_pred)
        if name == 'confusion_matrix':
            return metrics.confusion_matrix(y_true, y_pred)

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

    def mask_predict(self):
        X, y = [], []
        for file1 in tqdm(os.listdir('Labels\\RGB_superpixels'), total=len(os.listdir('Labels\\RGB_superpixels'))):
            for file2 in os.listdir('Labels\\mask_ground_truth'):
                print(file1.split('_')[-1], file2.split('_')[-1])
                print(file1[:file1.rindex('_')], file2[:file2.rindex('_')])
                if file1 in []:
                    pass
                if file1[:file1.rindex('_')] == file2[:file2.rindex('_')] and file1.split('_')[-1] == 'multiclass.npy':
                    spxl = np.load(os.path.join('Labels\\RGB_superpixels', file1))
                    R = spxl[:, :, 0].flatten()
                    G = spxl[:, :, 1].flatten()
                    B = spxl[:, :, 2].flatten()
                    for r, g, b in zip(R, G, B):
                        X.append([r, g, b])
                    y = np.load(os.path.join('Labels\\mask_ground_truth', file2)).flatten()
                    print(len(X), len(y))
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                    print('Random forest')
                    rfc = self.models['random_forest'].fit(X_train, y_train)
                    print('f1_score: ', self.metrics('f1_score', y_test, rfc.predict(X_test)))
                    print('precision: ', self.metrics('precision', y_test, rfc.predict(X_test)))
                    print('recall: ', self.metrics('recall', y_test, rfc.predict(X_test)))
                    print('accuracy', metrics.accuracy_score(y_test, rfc.predict(X_test)))
                    print(y_test)

    def classifier(self):
        df = pd.read_csv(self.file)

        names = df['name'].values
        X = df.iloc[:, 1:5].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # print('Random forest')
        # rfc = self.models['random_forest'].fit(X_train, y_train)
        # print('f1_score: ', self.metrics('f1_score', y_test, rfc.predict(X_test)))
        # print('precision: ', self.metrics('precision', y_test, rfc.predict(X_test)))
        # print('recall: ', self.metrics('recall', y_test, rfc.predict(X_test)))

        print('XGBoost')
        model = XGBClassifier()
        model.fit(X_train, y_train)
        print(metrics.accuracy_score(y_test, model.predict(X_test)))

        # print(self.metrics('confusion_matrix', y_test, rfc.predict(X_test)))
        # self.plot_cm(y_test, rfc.predict(X_test))
        # print(self.metrics('plot_confusion_matrix', y_test,
        #                    rfc.predict(X_test),
        #                    rfc,
        #                    X_test,
        #                    ['Bacterial_leaf_blight', 'Brown_spot', 'leaf_smut']))

    def NeuralNet(self):
        
        tfnet = model_tf.TfNet(self.file, 100, 128)
        model = tfnet.model()
        history = tfnet.train(model)
        # tfnet.plot(history)


    @functools.lru_cache(maxsize=128)
    def glcm(self):
        for file in tqdm(os.listdir(self.path), total=len(os.listdir(self.path))):
            if file.endswith('binary.npy'):
                image = np.load(os.path.join(self.path, file))

                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                glcm = greycomatrix(image, distances=[1], angles=[0], normed=True, symmetric=True)

                # Feature Extraction using GLCM
                self.name.append(file)
                self.contrast.append(greycoprops(glcm, 'contrast')[0][0])
                self.dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0][0])
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
    cl = Classify(file='features(binary_classify)(RGB).csv')
    # cl.glcm()
    # cl.classifier()
    # cl.make_label()
    # cl.mask_predict()
    # cl.make_json()
    nn = memory.cache(cl.NeuralNet())

