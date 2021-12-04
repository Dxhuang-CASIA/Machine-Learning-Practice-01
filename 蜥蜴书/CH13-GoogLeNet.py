import keras
import pandas as pd
from tensorflow.keras import Sequential, optimizers, layers, datasets
import tensorflow as tf
from matplotlib import pyplot as plt

class Inception(keras.Model):
    def __init__(self, c1, c2, c3, c4): # 输出通道
        super().__init__()
        self.p1_1 = layers.Conv2D(filters = c1, kernel_size = 1, activation = 'relu')

        self.p2_1 = layers.Conv2D(filters = c2[0], kernel_size = 1, activation = 'relu')
        self.p2_2 = layers.Conv2D(filters = c2[1], kernel_size = 3, padding = 'same', activation = 'relu')

        self.p3_1 = layers.Conv2D(filters = c3[0], kernel_size=1, activation='relu')
        self.p3_2 = layers.Conv2D(filters = c3[1], kernel_size = 5, padding = 'same', activation = 'relu')

        self.p4_1 = layers.MaxPool2D(pool_size = 3, strides = 1, padding = 'same')
        self.p4_2 = layers.Conv2D(c4, kernel_size = 1, activation = 'relu')

    def call(self, inputs):
        p1 = self.p1_1(inputs)
        p2 = self.p2_2(self.p2_1(inputs))
        p3 = self.p3_2(self.p3_1(inputs))
        p4 = self.p4_2(self.p4_1(inputs))
        return layers.Concatenate()([p1, p2, p3, p4])

def b1():
    return Sequential([
        layers.Conv2D(filters = 64, kernel_size = 7, strides = 2, padding = 'same', activation = 'relu'),
        layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')
    ])

def b2():
    return Sequential([
        layers.Conv2D(filters = 64, kernel_size = 1, activation = 'relu'),
        layers.Conv2D(filters = 192, kernel_size = 3, padding = 'same', activation = 'relu'),
        layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')
    ])

def b3():
    return Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')
    ])

def b4():
    return Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')
    ])

def b5():
    return Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        layers.GlobalAvgPool2D(),
        layers.Flatten()
    ])

def net():
    return Sequential([layers.Resizing(224, 224), b1(), b2(), b3(), b4(), b5(), layers.Dense(10, activation = 'softmax')])

def preProcess(data):
    data_mean = data.mean(axis = 0, keepdims = True)
    data_std = data.std(axis = 0, keepdims = True)
    data = (data - data_mean) / data_std
    return data

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