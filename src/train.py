import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np
from models import MusicTaggerCNN
from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from utils import read_hdf5

TF_WEIGHTS_PATH = "weights/music_tagger_cnn_weights_tensorflow.h5"


def train(data, epochs, batch_size=100, lr=0.001):
    net = MusicTaggerCNN()
    net.build((1, 96, 1366, 1))
    net.load_weights(TF_WEIGHTS_PATH, by_name=True)
    net.summary()

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    epoch_size = data["training"].__len__()

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        print("Epoch {}/{}".format((epoch + 1), epochs))
        progbar = tf.keras.utils.Progbar(epoch_size)
        for i in range(epoch_size):
            img, target = data["training"][i]
            with tf.GradientTape() as tape:
                output = net(img)
                loss = loss_fn(target, output)
                gradients = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net.trainable_variables))
                epoch_loss_avg.update_state(loss)
                epoch_accuracy.update_state(target, output)
                progbar.update(i + 1)
        print(
            "Epoch {}: Loss: {}, Accuracy: {}".format(
                epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()
            )
        )


def prepare_data(path="features/data.h5", test_size=0.1):
    data = read_hdf5(path)
    X, y = data["X"], data["y"]
    indices = np.arange(len(y))
    X_train, X_val, y_train, y_val, i_train, i_val = train_test_split(
        X, y, indices, test_size=test_size, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val, i_train, i_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val, i_train, i_val = prepare_data()
    data = {
        "training": DataGenerator(path="features/", list_IDs=i_train, shuffle=True),
        "valid": DataGenerator(path="features/", list_IDs=i_val, shuffle=False),
    }

    train(data, epochs=50)

