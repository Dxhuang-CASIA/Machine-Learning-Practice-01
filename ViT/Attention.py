import tensorflow as tf
from tensorflow.keras import layers
import math

class Self_Attention(tf.keras.Model):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    def __init__(self, dim_d, dim_v):
        super(Self_Attention, self).__init__()
        self.q = layers.Dense(dim_d)
        self.k = layers.Dense(dim_d)
        self.v = layers.Dense(dim_v)
        self._norm_fact = 1 / math.sqrt(dim_v)

    def call(self, x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x)
        V = self.v(x)
        atten = tf.nn.softmax(Q @ tf.transpose(K, (0, 2, 1)) * self._norm_fact)
        output = atten @ V
        return output

class Self_Attention_Multi_Head(tf.keras.Model):
    def __init__(self, dim_k, dim_v, nums_head):
        super(Self_Attention_Multi_Head, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = layers.Dense(dim_k)
        self.k = layers.Dense(dim_k)
        self.v = layers.Dense(dim_v)

        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / math.sqrt(dim_k)

    def call(self, x):
        Q = tf.reshape(self.q(x), (-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head))
        K = tf.reshape(self.k(x), (-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head))
        V = tf.reshape(self.v(x), (-1, x.shape[0], x.shape[1], self.dim_v // self.nums_head))

        atten = tf.nn.softmax(Q @ tf.transpose(K, perm = (0, 1, 3, 2)) * self._norm_fact)  # Q * K.T() # batch_size * seq_len * seq_len

        output = tf.reshape(atten @ V, (x.shape[0], x.shape[1], -1))  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output

model = Self_Attention_Multi_Head(4, 6, 2)
X = tf.random.normal((4, 3, 2))
print(model(X))