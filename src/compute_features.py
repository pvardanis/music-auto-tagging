import os
import multiprocessing
import tqdm
from scipy import stats
import pandas as pd
import librosa
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle
import argparse
from audio_feature_extractor import AudioFeatureExtractor, Config
from utils import *


def data_to_df(dataset_path):
    df = pd.DataFrame(columns=["filename", "label"])
    mapping = {}
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            mapping[str(i - 1)] = semantic_label
            filenames = [os.path.join(dirpath, f) for f in filenames]
            df = df.append(
                pd.DataFrame(
                    {"filename": filenames, "label": len(filenames) * [i - 1]}
                ),
                ignore_index=True,
            )

    return df, mapping


def save_data(filenames, label_mapping, X, y, output_dir="./data"):
    """Saves images & labels to hdf5.

    Args:
        X (np.array): mel spectrograms
        y (np.array): labels
        output_dir (str, optional): path to save hdf5 file. Defaults to "./data".
    """
    init_directory(output_dir)
    with h5py.File(f"{output_dir}/data.h5", "w") as hdf5:
        hdf5.create_dataset("mel_spectrograms", np.shape(X), dtype="float32", data=X)
        hdf5.create_dataset("labels", np.shape(y), dtype="int32", data=y)

    with open(f"{output_dir}/filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)

    with open(f"{output_dir}/labels.json", "w") as f:
        json.dump(label_mapping, f)


def afc_wrapper(index):
    audio = load_audio(df.filename[index], afc.sr, afc.n_samples)
    img = afc.mel_spectrogram(audio)
    return img, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", default="../genres/")
    parser.add_argument("--features_path", default="features/")
    parser.add_argument("--audio_duration", default=29.12, type=float)
    parser.add_argument("--sr", default=12000, type=int)
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument("--n_mels", default=96, type=int)
    parser.add_argument("--n_mfcc", default=13, type=int)
    parser.add_argument("--mfcc", dest="mfcc", action="store_true")
    parser.set_defaults(mfcc=False)

    args = parser.parse_args()

    assert os.path.isdir(args.audio_path), "Wrong audio path!"

    global df  # process in parallel
    global afc
    df, mapping = data_to_df(args.audio_path)

    cfg = Config(
        sr=args.sr,
        audio_duration=args.audio_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )
    afc = AudioFeatureExtractor(cfg)

    idx = [i for i in range(df.shape[0])]
    filenames = []
    X = []
    y = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for img, index in tqdm.tqdm(
        pool.imap_unordered(afc_wrapper, idx, chunksize=10), total=df.shape[0]
    ):
        X.append(img)
        y.append(df.label[index])
        filenames.append(df.filename[index])
    X = np.array(X)
    y = np.array(y)
    save_data(filenames, mapping, X, y, args.features_path)
