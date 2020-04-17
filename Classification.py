from joblib import Memory, Parallel, delayed
import joblib
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
from skopt import BayesSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv

from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm
import functools
import time
import warnings

# import model_tf

warnings.filterwarnings('ignore')

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
        self.binary_labels = np.load(
            'Labels\\RGB_superpixels\\binary_labels(RGB).npy')
        self.multiclass_labels = np.load(
            'Labels\\RGB_superpixels\\multiclass_labels(RGB).npy')

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
        for file1 in tqdm(os.listdir('Labels\\RGB_superpixels'), total=len(os.listdir('Labels\\RGB_superpixels')),
                          desc='RGB Superpixels'):
            for file2 in tqdm(os.listdir('Labels\\mask_ground_truth'), desc='Ground truths'):

                if file1 not in ['.gitignore', 'binary_labels(RGB).npy',
                                 'multiclass_labels(RGB).npy'] and file2 not in ['binary_labels.json',
                                                                                 'multiclass_labels.json',
                                                                                 '.gitignore']:
                    if file1[:file1.rindex('_')] == file2[:file2.rindex('_')] and file1.split('_')[-1] == 'multiclass.npy':

                        print(file1.split('_')[-1], file2.split('_')[-1])

                        spxl = np.load(os.path.join('Labels\\RGB_superpixels', file1))
                        R = spxl[:, :, 0].flatten()
                        G = spxl[:, :, 1].flatten()
                        B = spxl[:, :, 2].flatten()
                        for r, g, b in zip(R, G, B):
                            X.append([r, g, b])
                        y = np.load(os.path.join(
                            'Labels\\mask_ground_truth', file2)).flatten()
                        # print(len(X), len(y))

                        print(X)
                        exit(0)

    def bayes_optimization(self, X_train, y_train, is_binary=True):
        if is_binary:
            name = 'binary'
            objective = 'binary:logistic'
            eval_metric = 'auc'
            scoring = 'roc_auc'
            m_name = 'roc_auc'
        else:
            name = 'multiclass'
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            scoring = metrics.make_scorer(metrics.f1_score, average='weighted')
            m_name = 'f1'

        bayes_cv_tuner = BayesSearchCV(
            estimator=xgb.XGBClassifier(
                n_jobs=1,
                objective=objective,
                eval_metric=eval_metric,
                silent=1,
                tree_method='approx',
                n_estimators=200
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'max_depth': (1, 50),
                'n_estimators': (50, 100),
            },
            scoring=scoring,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            ),
            n_jobs=3,
            n_iter=10,
            verbose=3,
            refit=True,
            random_state=42
        )

        def status_print(optim_result):
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print('Model #{}\nBest {}: {}\nBest params: {}\n'.format(
                len(all_models),
                m_name,
                np.round(bayes_cv_tuner.best_score_, 4),
                bayes_cv_tuner.best_params_
            ))

            # Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            all_models.to_csv('results\\' + clf_name + f"({name})_cv_results.csv")

        return bayes_cv_tuner.fit(X_train, y_train, callback=status_print)

    def classifier(self, file, cls_name):
        df = pd.read_csv(file)

        names = df['name'].values
        X = df.iloc[:, 1:5].values
        y = df['label'].values

        X_resampled, y_resampled = Imbalance()(
            'repeated_edited_nearest_neighbours', X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25,
                                                            random_state=42)

        print(f'Using repeated_edited_nearest_neighbours')

        if cls_name == 'B':
            best_model = self.bayes_optimization(X_train, y_train)
        else:
            best_model = self.bayes_optimization(X_train, y_train, False)

        print('Accuracy: ', ClassificationMetrics()('accuracy', y_test, best_model.best_estimator_.predict(X_test)))
        self.plot_cm(y_test, best_model.predict(X_test))

    def NeuralNet(self):

        tfnet = model_tf.TfNet('features(binary_classify)(RGB).csv', 200, 32)
        model = tfnet.model()
        history = tfnet.train(model)
        # tfnet.plot(history)

    def mask_predict(self):
        X, y = [], []
        for file1 in tqdm(os.listdir('Labels\\RGB_superpixels')):
            for file2 in tqdm(os.listdir('Labels\\mask_ground_truth')):

                if file1 not in ['.gitignore', 'binary_labels(RGB).npy',
                                 'multiclass_labels(RGB).npy'] and file2 not in ['binary_labels.json',
                                                                                 'multiclass_labels.json',
                                                                                 '.gitignore']:
                    if file1[:file1.rindex('_')] == file2[:file2.rindex('_')] \
                            and file1.split('_')[-1] == 'multiclass.npy':
                        print(file1, file2)

                        # print(file1[:file1.rindex('_')], file2[:file2.rindex('_')])
                        # print(file1.split('_')[-1], file2.split('_')[-1])

                        spxl = np.load(os.path.join('Labels\\RGB_superpixels', file1))
                        label = np.load(os.path.join('Labels\\mask_ground_truth', file2))
                        print(spxl.shape, label.shape)

                        R = spxl[:, :, 0].flatten()
                        # print(len(R))

                        G = spxl[:, :, 1].flatten()
                        B = spxl[:, :, 2].flatten()
                        for r, g, b in zip(R, G, B):
                            X.append(r)
                        y = np.load(os.path.join(
                            'Labels\\mask_ground_truth', file2)).flatten()
                        # print(len(X), len(y))

                        # X_resampled, y_resampled = Imbalance()(
                        #     'repeated_edited_nearest_neighbours', X, y)

                        print('Before sampling: ', len(X), len(y))
                        X_resampled, y_resampled = Imbalance()(
                            'random_under_sampler', R.reshape(-1, 1), y)

                        print('Resampling done !')

                        X_train, X_test, y_train, y_test = train_test_split(
                            R.reshape(-1, 1), y, test_size=0.25, random_state=42)

                        # X_train, X_test, y_train, y_test = train_test_split(
                        #     X_resampled, y_resampled, test_size=0.25, random_state=42)
                        print('Splitting done ')

                        # best_model = self.bayes_optimization(X_train, y_train)
                        model = linear_model.LogisticRegression().fit(X_train, y_train)
                        # model = linear_model.LogisticRegression().fit(X_train, y_train)
                        print('Accuracy: ',
                              ClassificationMetrics()('accuracy', y_test, model.predict(X_test)))

    @functools.lru_cache(maxsize=128)
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
    cl = Classify()
    # cl.glcm()
    # cls_name = input('Enter the type of classification from [B] or [M]: ')
    # if cls_name == 'B':
    #     cl.classifier(binary_file, cls_name)
    # else:
    #     cl.classifier(multiclass_file, cls_name)
    # cl.make_label()
    cl.mask_predict()
    # cl.make_json()
    # cl.NeuralNet()
