import keras
from keras.models import load_model
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

model = load_model('keras_models/model(multiclass).h5')

print(model.summary())
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

score = model.evaluate(X_test, y_test)

# print(f'Precison: {precision_score(y_test, model.predict(X_test), average="weighted")}')
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
