import tensorflow as tf
import numpy as np
import os
import pickle
from utils import read_hdf5


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        path,
        list_IDs,
        batch_size=64,
        dim=(96, 1366, 1),
        validation_split=0.9,
        is_training=True,
        shuffle=True,
    ):
        self.list_IDs = list_IDs

        self.data = read_hdf5(path + "data.h5")
        with open(path + "filenames.pkl", "rb") as fp:
            self.filenames = pickle.load(fp)

        self.dim = dim
        self.batch_size = batch_size

        self.validation_split = validation_split
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch = self.__data_generation(ids)
        assert batch[0].size != 0 and batch[1].size != 0
        return batch

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.array([self.data["X"][i] for i in list_IDs_temp])
        X = X[..., np.newaxis]
        y = np.array([self.data["y"][i] for i in list_IDs_temp])

        return X, y

    def on_epoch_end(self):
        self.indexes = [i for i in self.list_IDs]
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

