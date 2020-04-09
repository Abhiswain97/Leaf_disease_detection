import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_csv('features(multiclass_classify).csv')

X = data.iloc[:, 1:6]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# print(X_train)
model = Sequential(
    [Dense(16, input_dim=5, activation='relu'),
     Dense(32, activation='relu'),
     Dense(128, activation='relu'),
     Dense(64, activation='relu'),
     Dense(3, activation='softmax')]
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
csv_logger = CSVLogger('training.csv', separator=',')
history = model.fit(X_train, y_train,
                    validation_split=0.33,
                    epochs=50,
                    batch_size=16,
                    verbose=1,
                    shuffle=True,
                    callbacks=[csv_logger])
print(history.history.keys())

model_json = model.to_json()

model.save('keras_models/model(multiclass).h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
