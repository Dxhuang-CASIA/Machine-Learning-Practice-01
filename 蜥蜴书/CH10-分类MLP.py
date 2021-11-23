import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# 模型 MLP 数据 Fashion MNIST 分类模型

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# 创建验证集
X_valid, X_train = X_train_full[: 5000] / 255.0, X_train_full[5000: ] / 255.
y_valid, y_train = y_train_full[: 5000], y_train_full[5000: ]

# 创建标签
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 创建网络
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# print('---------------------------------模型信息---------------------------------')
# print(model.summary()) # 输出模型信息
#
# print('---------------------------------访问特定的层---------------------------------')
# print(model.layers[1].name)
#
# print('---------------------------------查看各层的参数---------------------------------')
# print(model.layers[1].get_weights()) #weights, bias

# 编译模型
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd", metrics = ["accuracy"]) # 因为不是one-hot编码 如果是one-hot loss用categorical_crossentropy

# 转化为one-hot keras.utils.to_categorical()

# 训练和评估模型
histroy = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

pd.DataFrame(histroy.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)

# 使用模型进行预测
X_new = X_test[:3]
y_prob = model.predict(X_new)
print(y_prob.round(2))

y_pred = np.argmax(y_prob, axis = 1)
print(y_pred)
np.array(['Ankle boot', 'Pullover', 'Trouser'], dtype = '<U11')

y_new = y_test[:3]
print(y_new)