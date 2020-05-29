from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import numpy as np

data = pd.read_csv("features(disease).csv")
print(data.head())

data["label"] = data["label"].astype("category").cat.codes

labels_encoded = pd.get_dummies(data["label"])
# print(labels_encoded)
data.drop(["label"], axis=1, inplace=True)
# print(data.head())

feature_df = pd.concat([data, labels_encoded], axis=1)
# print(feature_df.head())

X = feature_df.iloc[:, :4]
X = X / np.mean(X)

# print(X)
# exit(0)
y = feature_df.iloc[:, 5:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print(clf)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(clf.feature_importances_)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)
# print(clf.predict_proba(X_test))
