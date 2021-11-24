import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y):
    x = tf.cast(x, dtype = tf.float32) / 255.
    y = tf.cast(y, dtype = tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
# print(x.shape, y.shape) # (60000, 28, 28) (60000,)
batchSize = 128

db = tf.data.Dataset.from_tensor_slices((x, y)) # 构造数据集
db = db.map(preprocess).shuffle(10000).batch(batchSize)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchSize)

db_iter = iter(db)
sample = next(db_iter)
# print('batch: ', sample[0].shape, sample[1].shape)

model = Sequential([layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',padding='same'),
                    layers.AvgPool2D(pool_size=2, strides=2),
                    layers.Conv2D(filters=16, kernel_size=5,activation='sigmoid'),
                    layers.AvgPool2D(pool_size=2, strides=2),
                    layers.Flatten(),
                    layers.Dense(120, activation='sigmoid'),
                    layers.Dense(84, activation='sigmoid'),
                    layers.Dense(10)])
model.build(input_shape = [None, 28, 28, 1])
model.summary()
optimizer = optimizers.Adam(lr = 1e-3)
def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            # x[b, 28, 28] y [b]
            x = tf.reshape(x, [-1, 28, 28, 1])
            with tf.GradientTape() as tape:
                logits = model(x) # forward
                y_onehot = tf.one_hot(y, depth = 10)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss2 = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits = True)
                loss2 = tf.reduce_mean(loss2)
            grads = tape.gradient(loss2, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss), float(loss2))
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28, 28, 1])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis = 1)
            pred = tf.argmax(prob, axis = 1)
            pred = tf.cast(pred, dtype = tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype = tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]
        acc = total_correct / total_num
        print('----------------------------')
        print(epoch, 'test acc: ', acc)


if __name__ == '__main__':
    main()