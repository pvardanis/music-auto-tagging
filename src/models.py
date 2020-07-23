import os
import time
import h5py

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np

tf.keras.backend.set_image_data_format("channels_last")

TF_WEIGHTS_PATH = "weights/data_music_tagger_cnn_weights_tensorflow.h5"


class MusicTaggerCNN(tf.keras.Model):
    def __init__(self, input_shape=(96, 1366, 1), trainable=False, include_top=True):
        super().__init__()

        self.include_top = include_top
        self.input_ = tf.keras.layers.InputLayer(input_shape=input_shape,)
        self.bn = keras.layers.BatchNormalization(
            axis=2, trainable=trainable, name="bn_0_freq"
        )  # [None, freq, time, channels]

        # conv block 1
        self.conv_1 = self.conv_block(32, (3, 3), 1, "same", (2, 4), trainable, "1")
        # conv block 2
        self.conv_2 = self.conv_block(128, (3, 3), 1, "same", (2, 4), trainable, "2")
        # conv block 3
        self.conv_3 = self.conv_block(128, (3, 3), 1, "same", (2, 4), trainable, "3")
        # conv block 4
        self.conv_4 = self.conv_block(192, (3, 3), 1, "same", (3, 5), trainable, "4")
        # conv block 5
        self.conv_5 = self.conv_block(256, (3, 3), 1, "same", (4, 4), True, "5")
        # output
        self.flatten = keras.layers.Flatten(name="flatten")
        self.out = keras.layers.Dense(10, activation="sigmoid", name="out")

    def call(self, inputs):
        x = self.bn(self.input_(inputs))
        for layer in self.conv_1:
            x = layer(x)
        for layer in self.conv_2:
            x = layer(x)
        for layer in self.conv_3:
            x = layer(x)
        for layer in self.conv_4:
            x = layer(x)
        for layer in self.conv_5:
            x = layer(x)
        x = self.out(self.flatten(x))
        return x

    def conv_block(
        self, filters, kernel_size, strides, padding, pool_size, trainable, name, batch_norm=True, 
    ):
        # block = keras.Sequential(name="conv_block_" + name)
        block = []
        block.append(
            keras.layers.Convolution2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                trainable=trainable,
                name="conv"+name
            )
        )
        # batch_norm
        if batch_norm:
            block.append(keras.layers.BatchNormalization(trainable=trainable,name="bn"+name))
        block.append(keras.layers.ELU(name="elu_"+name))
        block.append(keras.layers.MaxPooling2D(name="pool"+name))

        return block


class MusicTaggerCRNN(tf.keras.Model):
    def __init__(self, input_shape=(96, 1366, 1), include_top=True):
        super().__init__()

        self.include_top = include_top
        self.input_ = tf.keras.layers.InputLayer(input_shape=input_shape,)
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(0, 37))
        self.bn = keras.layers.BatchNormalization(
            axis=1, name="bn_0_freq"
        )  # [None, freq, time, channels]

        # conv block 1
        self.conv_1 = self.conv_block(64, (3, 3), 1, "same", (2, 2), "1")
        # conv block 2
        self.conv_2 = self.conv_block(128, (3, 3), 1, "same", (3, 3), "2")
        # conv block 3
        self.conv_3 = self.conv_block(128, (3, 3), 1, "same", (4, 4), "3")
        # conv block 4
        self.conv_4 = self.conv_block(128, (3, 3), 1, "same", (4, 4), "4")
        # reshape
        self.reshape = keras.layers.Reshape((15, 128))
        # GRUs
        self.gru_1 = keras.layers.GRU(32, return_sequences=True, name="gru1",)
        self.gru_2 = keras.layers.GRU(32, return_sequences=False, name="gru2",)
        # output
        self.dropout = keras.layers.Dropout(0.3)
        self.out = keras.layers.Dense(50, activation="sigmoid", name="output")

    def conv_block(
        self, filters, kernel_size, strides, padding, pool_size, name, batch_norm=True,
    ):
        block = keras.Sequential(name="conv_block_" + name)
        block.add(
            keras.layers.Convolution2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                name="conv_" + name,
            )
        )
        # batch_norm
        if batch_norm:
            block.add(keras.layers.BatchNormalization(name="bn" + name))
        block.add(keras.layers.ELU(name="elu_" + name))
        block.add(
            keras.layers.MaxPooling2D(
                strides=pool_size, pool_size=pool_size, name="pool" + name
            )
        )
        block.add(keras.layers.Dropout(0.1, name="dropout" + name))

        return block

    def call(self, inputs):
        x = self.bn(self.zero_padding(self.input_(inputs)))
        print(np.max(x), np.min(x))
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.reshape(x)
        x = self.gru_2(self.gru_1(x))
        x = self.dropout(x)
        if self.include_top:
            x = self.out(x)
        return x


# if __name__ == "__main__":
#     net = MusicTaggerCNN()
#     net.build((1, 96, 1366, 1))

#     net.load_weights(TF_WEIGHTS_PATH, by_name=True)
#     net.summary()
#     x = 5
