import os
import numpy as np
import librosa
import tensorflow as tf
import h5py


def init_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def amplitude_to_db(mag, amin=1 / (2 ** 16), normalize=True):
    mag_db = 20 * np.log1p(mag / amin)
    if normalize:
        mag_db /= 20 * np.log1p(1 / amin)
    return mag_db


def power_to_db(mag, amin=1 / (2 ** 16), normalize=True):
    mag_db = 10 * np.log1p(mag / amin)
    if normalize:
        mag_db /= 10 * np.log1p(1 / amin)
    return mag_db


def db_to_amplitude(mag_db, amin=1 / (2 ** 16), normalize=True):
    if normalize:
        mag_db *= 20 * np.log1p(1 / amin)
    return amin * np.expm1(mag_db / 20)


def compute_stft(audio, n_fft=2014, power=1, normalize=True):
    window = np.hanning(n_fft)
    S = librosa.stft(audio, n_fft=n_fft, hop_length=int(n_fft / 2), window=window)
    mag, phase = np.abs(S), np.angle(S)
    if normalize:
        mag = 2 * mag / np.sum(window)  # between 0 and 1
    return mag ** power, phase


def compute_istft(mag, phase, n_fft=2014, normalize=True):
    window = np.hanning(n_fft)
    if normalize:
        mag = mag * np.sum(window) / 2
    R = mag * np.exp(1j * phase)
    audio = librosa.stft(R, hop_length=int(n_fft / 2), window=window)
    return audio


def load_audio(filename, sr, n_samples):
    audio = librosa.core.load(filename, sr=sr)[0]
    if 0 < len(audio):  # workaround: 0 length causes error
        audio, _ = librosa.effects.trim(audio)
    if len(audio) >= n_samples:  # long enough
        audio = audio[0:n_samples]
    else:  # pad blank
        padding = n_samples - len(audio)
        offset = padding // 2
        audio = np.pad(audio, (offset, n_samples - len(audio) - offset), "constant")
    return audio[:n_samples]


def read_hdf5(path="features/data.h5"):
    with h5py.File(path, "r") as f:
        data = {"X": f["mel_spectrograms"][()], "y": f["labels"][()]}
    return data


def visitor_func(name, node):
    if isinstance(node, h5py.Group):
        print(node.name, "is a Group")
    elif isinstance(node, h5py.Dataset):
        if node.dtype == "object":
            print(node.name, "is an object Dataset")
        else:
            print(node.name, "is a Dataset")
    else:
        print(node.name, "is an unknown type")


def correct_weights(path):
    with h5py.File(path, "a") as f:
        for group in f.keys():
            print(group)
            if group.startswith("conv"):
                for dset in f[group].keys():
                    if "W:0" in dset:
                        arr = f[group][dset][:]
                        replace = np.reshape(
                            arr,
                            (arr.shape[1], arr.shape[0], arr.shape[2], arr.shape[3]),
                        )
                        del f[group][dset]
                        f[group].create_dataset(dset, data=replace)

