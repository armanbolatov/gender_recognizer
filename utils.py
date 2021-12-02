import pandas as pd
import numpy as np
import os, zipfile
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data(vector_length=140):
    """
    A function to load preprocessed dataset from the train folder. After
    the second run it retrieves the dataset from result/features.npy and
    results/labels.npy folders
    Input: vector length
    Output: features matrix and labels vector
    """
    # if results directory doesn't exist, create one
    if not os.path.isdir("results"):
        os.mkdir("results")

    # if the dataset is already in results folder, load them from there
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y

    # read the labels of the dataset from train_raw.zip/train/targets.tsv
    train_zip = zipfile.ZipFile('train_raw.zip', 'r')
    train_targets = train_zip.open('train/targets.tsv')
    df = pd.read_csv(train_targets, delimiter='\t', names=['filename', 'gender'])

    # initialize and fill the feature matrix and labels vector
    n_samples = len(df)
    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))
    dataset = enumerate(zip(df['filename'], df['gender']))
    for i, (filename, gender) in tqdm(dataset, "Loading", total=n_samples):
        features = np.load(f"train/{filename}.npy")
        X[i] = features
        y[i] = gender

    # save the audio features and labels into files
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y

def split_data(X, y, test_size=0.1, valid_size=0.1):
    """
    Input: features matrix and labels vector of the audiofiles
    Output: training, testing, and validation sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=7
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=7
    )
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

def create_model(vector_length=140):
    """
    Input: vector length
    Output: sequential neural network
    """
    model = Sequential([
        Dense(512, input_shape=(vector_length,)), Dropout(0.3),
        Dense(512, activation="relu"), Dropout(0.2),
        Dense(256, activation="relu"), Dropout(0.3),
        Dense(256, activation="relu"), Dropout(0.2),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(128, activation="relu"), Dropout(0.2),
        Dense(64, activation="relu"), Dropout(0.2),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    model.summary()
    return model
