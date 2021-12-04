import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, optimizers

(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:, np.newaxis]
y_train = tf.one_hot(y_train, depth = 10)
y_test = tf.one_hot(y_test, depth = 10)

model = Sequential([layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',padding='same'),
                    layers.AvgPool2D(pool_size=2, strides=2),
                    layers.Conv2D(filters=16, kernel_size=5,activation='sigmoid'),
                    layers.AvgPool2D(pool_size=2, strides=2),
                    layers.Flatten(),
                    layers.Dense(120, activation='sigmoid'),
                    layers.Dense(84, activation='sigmoid'),
                    layers.Dense(10, activation = 'softmax')])

model.build(input_shape = [None, 28, 28, 1])
model.summary()
optimizer = optimizers.Adam(lr = 1e-3)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history= model.fit(X_train, y_train, batch_size = 128, epochs = 20)
score = model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()