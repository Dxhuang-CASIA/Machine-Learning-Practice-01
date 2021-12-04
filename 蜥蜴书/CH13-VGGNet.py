import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, Sequential, layers, optimizers
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(layers.Conv2D(filters = num_channels, kernel_size = 3, padding = 'same', activation = 'relu'))
    blk.add(layers.MaxPool2D(pool_size = 2, strides = 2))
    return blk

def vgg_11(conv_arch):
    net = Sequential(layers.Resizing(224, 224))
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    net.add(Sequential([
        layers.Flatten(),
        layers.Dense(4096),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(4096),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10),
        layers.Activation('softmax')
    ]))
    return net

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

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
model = vgg_11(conv_arch)

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