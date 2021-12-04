# ResNet18 (4 * 4 + 1 + 1)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides = stride, padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides = 1, padding = 'same')
        self.bn2 = layers.BatchNormalization()

        if stride!= 1: # 下采样
            self.dowmsample = Sequential()
            self.dowmsample.add(layers.Conv2D(filter_num, (1, 1), strides = stride, padding = 'same'))
        else:
            self.dowmsample = lambda x : x

    def call(self, inputs, training = True):
        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.dowmsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes = 100): # layer_dims = [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 预处理层
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides = (1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size = (2, 2), strides = (1, 1), padding = 'same')])
        # 4个 ResBlock 每个里面又有2个BasicBlock 1个BasicBlock又是由conv->batchnorm->relu->conv->batchnorm构成的
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride = 2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride = 2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output : [b, 512, h, w] channel = 512
        self.avgpool = layers.GlobalAveragePooling2D() # 不管h和w多少 自适应
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x) # [b,channel]
        x = self.fc(x) # [b, 100]
        return x

    def build_resblock(self, filter_num, blocks, stride = 1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride = 1)) # not down sample
        return res_blocks

def ResNet18():

    return ResNet([2, 2, 2, 2], num_classes = 10)
