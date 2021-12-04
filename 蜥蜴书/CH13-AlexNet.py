import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, datasets, layers

def preProcess(data):
    data_mean = data.mean(axis = 0, keepdims = True)
    data_std = data.std(axis = 0, keepdims = True)
    data = (data - data_mean) / data_std
    return data

def net():
    net = Sequential()
    net.add(layers.Resizing(224, 224))
    net.add(layers.Conv2D(filters = 96, kernel_size = 11, strides = 4, activation = 'relu'))
    net.add(layers.MaxPool2D(pool_size = 3, strides = 2))
    net.add(layers.Conv2D(filters = 256, kernel_size = 5, padding = 'same', activation = 'relu'))
    net.add(layers.MaxPool2D(pool_size = 3, strides = 2))
    net.add(layers.Conv2D(filters = 384, kernel_size = 3, padding = 'same', activation = 'relu'))
    net.add(layers.Conv2D(filters = 384, kernel_size = 3, padding = 'same', activation = 'relu'))
    net.add(layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
    net.add(layers.MaxPool2D(pool_size = 3, strides = 2))
    net.add(layers.Flatten())
    net.add(layers.Dense(4096, activation = 'relu'))
    net.add(layers.Dropout(0.5))
    net.add(layers.Dense(4096, activation = 'relu'))
    net.add(layers.Dropout(0.5))
    net.add(layers.Dense(10, activation = 'softmax'))
    return net

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = preProcess(X_train)
X_test = preProcess(X_test)
y_train = tf.one_hot(y_train.squeeze(), depth = 10)
y_test = tf.one_hot(y_test.squeeze(), depth = 10)

model = net()
model.build(input_shape = [None, 32, 32, 3])
print(model.summary())

optimizer = optimizers.SGD(learning_rate = 1e-3, momentum = 0.9)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history= model.fit(X_train, y_train, batch_size = 128, epochs = 20)
score = model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()