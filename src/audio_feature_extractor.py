import librosa
import numpy as np
from utils import power_to_db, compute_stft


class Config:
    def __init__(
        self, sr, audio_duration, n_fft=1024, hop_length=512, n_mels=128, n_mfcc=13
    ):
        self.sr = sr
        self.audio_duration = audio_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_samples = int(sr * audio_duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc


class AudioFeatureExtractor:
    def __init__(self, config):
        self.audio_duration = config.audio_duration
        self.sr = config.sr
        self.n_samples = config.n_samples
        self.n_mels = config.n_mels
        self.n_mfcc = config.n_mfcc
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length

    def compute_stft(self, audio, power=2, normalize=True):
        return compute_stft(
            audio=audio, n_fft=self.n_fft, power=power, normalize=normalize
        )

    def mel_spectrogram(self, audio, power=2, normalize=True):
        # mag, _ = compute_stft(
        #     audio=audio, n_fft=self.n_fft, power=power, normalize=normalize
        # )
        # mel = librosa.feature.melspectrogram(
        #     S=mag,
        #     sr=self.sr,
        #     n_mels=self.n_mels,
        #     hop_length=self.hop_length,
        #     n_fft=self.n_fft,
        # )
        # mel_db = power_to_db(mel)
        logpow = librosa.power_to_db
        melgram = librosa.feature.melspectrogram
        mel_db = logpow(
            melgram(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            ),
            ref=1.0,
        )
        return mel_db

    def mfcc(self, audio):
        return librosa.feature.mfcc(
            audio,
            self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).T
