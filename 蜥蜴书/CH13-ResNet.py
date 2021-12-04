import tensorflow as tf
import numpy as np
import pandas as pd
import loda_data
from tensorflow.keras import layers, datasets, Sequential, activations, optimizers, applications
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Residual(tf.keras.Model):
    def __init__(self, num_channel, use_conv1x1 = False, strides = 1): # 只要输入通道跟num_channel不同或者strides大于1就得使用True
        super().__init__()
        self.conv_1 = layers.Conv2D(filters = num_channel, kernel_size = 3, strides = strides, padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.conv_2 = layers.Conv2D(filters = num_channel, kernel_size = 3, padding = 'same')
        self.bn2 = layers.BatchNormalization()
        self.conv_3 = None
        if use_conv1x1:
            self.conv_3 = layers.Conv2D(filters = num_channel, kernel_size = 1, strides = strides)

    def call(self, inputs):
        y = self.relu1(self.bn1(self.conv_1(inputs)))
        y = self.bn2(self.conv_2(y))
        if self.conv_3 is not None:
            inputs = self.conv_3(inputs)
        y += inputs
        return activations.relu(y)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block = False):
        super().__init__()
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(Residual(num_channels, use_conv1x1 = True, strides = 2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, input):
        for layer in self.residual_layers:
            input = layer(input)
        return input

def net():
    return Sequential([ # [224, 224, 3]
        tf.keras.layers.Resizing(224, 224),
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same'), # [112, 112, 64]
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'), # [56, 56, 64]
        ResBlock(64, 2, first_block=True), # [56, 56, 64]
        ResBlock(128, 2), # [28, 28, 128]
        ResBlock(256, 2), # [14, 14, 256]
        ResBlock(512, 2), # [7, 7, 512]
        layers.GlobalAvgPool2D(), # [1, 512]
        layers.Dense(10), # [1, 10]
        layers.Activation('softmax')
    ])

(X_train, y_train), (X_test, y_test) = loda_data.cifar10()
X_valid, X_train = X_train[: 5000], X_train[5000: ]
y_valid, y_train = y_train[: 5000], y_train[5000: ]

model = net()
model.build(input_shape = [None, 224, 224, 3])
print(model.summary())

optimizer = optimizers.SGD(learning_rate = 1e-3, momentum = 0.9)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history= model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_valid, y_valid))
score = model.evaluate(X_test, y_test)