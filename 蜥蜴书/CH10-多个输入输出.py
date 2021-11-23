from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd

# 多个输入
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full) # 默认test_size = 0.25

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 划分两个输入
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]

# 构建网络
input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = "relu")(input_B)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat = keras.layers.concatenate([input_A, input_B])
# output = keras.layers.Dense(1, name = "output")(concat)
# model = keras.Model(inputs = [input_A, input_B], outputs = [output])

output = keras.layers.Dense(1, name = "main_output")(concat)
aux_output = keras.layers.Dense(1, name = "aux_output")(hidden2)
model = keras.Model(inputs = [input_A, input_B], outputs = [output, aux_output])

# 训练
# model.compile(loss = "mean_squared_error", optimizer = "sgd")
model.compile(loss = ["mse", "mse"], loss_weights = [0.9, 0.1], optimizer = keras.optimizers.SGD(learning_rate=1e-3))

# history = model.fit((X_train_A, X_train_B), y_train, epochs = 20, validation_data = ((X_valid_A, X_valid_B), y_valid))
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs = 20, validation_data = ([X_valid_A, X_valid_B], [y_valid, y_valid]))
# mse_test = model.evaluate((X_test_A, X_test_B), y_test)
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])

plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()