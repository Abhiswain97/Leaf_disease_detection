import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from handle_imbalance import Imbalance

import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras.callbacks import EarlyStopping

wandb.init(project="leaf_disease")


class TfNet:
    def __init__(self, file, epochs, batch_size):
        self.data = pd.read_csv(file)
        self.epochs = epochs
        self.batch_size = batch_size

    def train_test_split(self):
        X = self.data.iloc[:, 1:6]
        y = self.data["label"]

        X_resampled, y_resampled = Imbalance()(
            "repeated_edited_nearest_neighbours", X, y
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        # np.save('X_test.npy', X_test)
        # np.save('y_test.npy', y_test)
        return X_train, X_test, y_train, y_test

    def model(self):
        model = Sequential(
            [
                Dense(16, input_dim=5, activation="relu"),
                Dense(32, activation="relu"),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )
        return model

    def train(self, model):
        X_train, X_test, y_train, y_test = self.train_test_split()
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["sparse_categorical_accuracy"],
        )
        csv_logger = CSVLogger("training.csv", separator=",")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            shuffle=True,
            callbacks=[
                csv_logger,
                WandbCallback(),
                EarlyStopping(monitor="val_loss", mode="min", verbose=1),
            ],
        )
        # print(history.history.keys())

        # model_json = model.to_json()

        # model.save('keras_models/model(multiclass).h5')
        model.save(os.path.join(wandb.run.dir, "model.h5"))

        return history

    def plot(self, history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()
