import pandas as pd
import librosa
import zipfile
from fnmatch import fnmatch
import numpy as np
from tqdm import tqdm

def extract_feature(X, sr):
    """
    Extracts numerical features of a wave signal
    Input: a wave signal and a sample rate
    Output: a numpy matrix with the signal's melspectrogram and chromagram
    """
    result = np.array([])
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)
    log_mel = librosa.amplitude_to_db(mel)
    result = np.hstack((result, log_mel))
    return result

def prepare(zip_name):
    """
    Extracts features of .wav files in a zip and save them in .npy format
    Input: name of the .zip file
    Output: nothing
    """
    zip = zipfile.ZipFile(zip_name, 'r')
    for file in tqdm(zip.namelist()):
        if fnmatch(file, '*.wav'):
            signal, sr = librosa.load(zip.open(file))
            features = extract_feature(signal, sr)
            np.save(file[:-4], features)

prepare('train_raw.zip')
prepare('test_raw.zip')
