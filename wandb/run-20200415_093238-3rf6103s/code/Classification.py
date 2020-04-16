import os
import pickle
import json
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import metrics

from metrics import ClassificationMetrics
from handle_imbalance import Imbalance

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import linear_model, tree, ensemble, metrics
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv

from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm
import functools
import time
import warnings

# import model_tf

import wandb
wandb.init()

warnings.filterwarnings('ignore')
import joblib
from joblib import Memory, Parallel, delayed

location = 'cachedir'
memory = Memory(location, verbose=0)


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
        self.binary_labels = np.load('Labels\\RGB_superpixels\\binary_labels(RGB).npy')
        self.multiclass_labels = np.load('Labels\\RGB_superpixels\\multiclass_labels(RGB).npy')

    def create_folds(self, file):
        df = pd.read_csv(file)
        df["kfold"] = -1

        df = df.sample(frac=1).reset_index(drop=True)

        kf = StratifiedKFold(n_splits=5)

        for fold, (trn_, val_) in enumerate(kf.split(X = df.iloc[:, 1:5].values, y=df['label'].values)):
            print(len(trn_), len(val_))
            df.loc[val_, 'kfold'] = fold
    
        df.to_csv("train_folds.csv", index=False)
        

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

    def mask_predict(self):
        X, y = [], []
        for file1 in tqdm(os.listdir('Labels\\RGB_superpixels'), total=len(os.listdir('Labels\\RGB_superpixels'))):
            for file2 in os.listdir('Labels\\mask_ground_truth'):
                print(file1.split('_')[-1], file2.split('_')[-1])
                print(file1[:file1.rindex('_')], file2[:file2.rindex('_')])
                if file1[:file1.rindex('_')] == file2[:file2.rindex('_')] and file1.split('_')[-1] == 'multiclass.npy':
                    spxl = np.load(os.path.join('Labels\\RGB_superpixels', file1))
                    R = spxl[:, :, 0].flatten()
                    G = spxl[:, :, 1].flatten()
                    B = spxl[:, :, 2].flatten()
                    for r, g, b in zip(R, G, B):
                        X.append([r, g, b])
                    y = np.load(os.path.join('Labels\\mask_ground_truth', file2)).flatten()
                    print(len(X), len(y))

                    X_resampled, y_resampled = Imbalance()('repeated_edited_nearest_neighbours', X, y)

                    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)
                    print('Random forest')
                    rfc = self.models['random_forest'].fit(X_train, y_train)
                    print('f1_score: ', self.metrics('f1_score', y_test, rfc.predict(X_test)))
                    print('precision: ', self.metrics('precision', y_test, rfc.predict(X_test)))
                    print('recall: ', self.metrics('recall', y_test, rfc.predict(X_test)))
                    print('accuracy', metrics.accuracy_score(y_test, rfc.predict(X_test)))
                    print(y_test)

    def classifier(self, file):
        df = pd.read_csv(file)

        names = df['name'].values
        X = df.iloc[:, 1:5].values
        y = df['label'].values

        X_resampled, y_resampled = Imbalance()('repeated_edited_nearest_neighbours', X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25,
                                                            random_state=42)

        print(f'Using repeated_edited_nearest_neighbours')

        for model in ['CatBoost', 'XGBoost']:
            if model == 'Cat':
                print('CatBoost')
                cat_model = CatBoostClassifier(
                    iterations=1000, 
                    learning_rate=0.01, 
                    loss_function='Logloss', 
                    eval_metric='Accuracy',
                    use_best_model=True
                )
                cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))

                self.plot_cm(y_test, cat_model.predict(X_test))

            if model == 'XGBoost':

                print('XGBoost')
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': ['logloss','auc'],
                    'num_class': 2,
                    'max_depth': 6
                } 

                train_data = xgb.DMatrix(X_train, label=y_train)
                test_data = xgb.DMatrix(X_test, label=y_test)

                evallist = [(test_data, 'eval'), (train_data, 'train')]

                bst = xgb.train(
                    params, 
                    train_data, 
                    10,
                    evallist,  
                    callbacks=[wandb.xgboost.wandb_callback()]
                )

                # model = XGBClassifier()
                # model.fit(X_train, y_train)
                
                # metric = ClassificationMetrics()
                # print('Accuracy score: ', metric('accuracy', y_test, model.predict(X_test)))
                # print('Precision score: ', metric('precision', y_test, model.predict(X_test)))
                # print('Recall score: ', metric('recall', y_test, model.predict(X_test)))
                # print('Log loss: ', metric('logloss', y_test, None, model.predict_proba(X_test)))
                # print('f1 score: ', metric('f1', y_test, model.predict(X_test)))
                
                # timestampTime = time.strftime("%H:%M:%S")
                # timestampDate = time.strftime("%d/%m/%Y")
                # timestampEND = timestampDate + '-' + timestampTime
                
                # results = {
                #     'model': 'XGBoost',
                #     'accuracy': metric('accuracy', y_test, model.predict(X_test)),
                #     'Precision score: ': metric('precision', y_test, model.predict(X_test)),
                #     'Recall score: ': metric('recall', y_test, model.predict(X_test)),
                #     'f1 score': metric('f1', y_test, model.predict(X_test)),
                #     'under_sampler': 'repeated_edited_nearest_neighbours',
                #     'time_stamp': timestampEND
                # }
                
                # fname = file.split('.')[0]
                
                # with open(f'results\\{fname}-6.json', 'w') as fp:
                #     json.dump(results, fp)

                self.plot_cm(y_test, model.predict(X_test))

    def NeuralNet(self):

        tfnet = model_tf.TfNet('features(binary_classify)(RGB).csv', 200, 32)
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
    cl = Classify()
    # cl.glcm()
    cl.classifier('features(binary_classify)(RGB).csv')
    # cl.make_label()
    # cl.mask_predict()
    # cl.make_json()
    # cl.NeuralNet()
    # cl.create_folds('features(binary_classify)(RGB).csv')
