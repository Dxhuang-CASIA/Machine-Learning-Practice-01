import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from ResNet import ResNet18

(x, y), (x_val, y_val) = datasets.cifar10.load_data() # (50000, 32, 32, 3)
y = tf.squeeze(y, axis = 1)
y_val = tf.squeeze(y_val, axis = 1)
y = tf.one_hot(y, depth = 10)
y_val = tf.one_hot(y_val, depth = 10)

def main():
    model = ResNet18()
    model.build(input_shape = (None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr = 1e-4)

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(x, y, batch_size=128, epochs=10)
    score = model.evaluate(x_val, y_val)

if __name__ == '__main__':
    main()