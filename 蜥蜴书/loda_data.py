from tensorflow.keras import datasets
import tensorflow as tf

def preProcess(data):
    data_mean = data.mean(axis = 0, keepdims = True)
    data_std = data.std(axis = 0, keepdims = True)
    data = (data - data_mean) / data_std
    return data

def cifar10():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train = preProcess(X_train)
    X_test = preProcess(X_test)
    y_train = tf.one_hot(y_train.squeeze(), depth=10)
    y_test = tf.one_hot(y_test.squeeze(), depth=10)
    return (X_train,y_train), (X_test, y_test)