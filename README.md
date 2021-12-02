# Gender Recognition from Voice via Librosa

This is a solution for the [Biometry task](https://contest.yandex.ru/contest/28413/problems/F/) in the Yandex ML training contest.

## Problem statement

In this task, you need to predict the gender of the person whose speech is recorded on each of the files using sound files in `.wav` format (0 — man, 1 — woman).

In order to complete this problem, you need to get an accuracy of more than 98 percent on the **[test dataset](https://yadi.sk/d/K8Z-_gQbspmxkhw)**.

The **[training dataset](https://yadi.sk/d/IUUTPJFOfwn_OQ)** has a `targets.tsv` file that contains the correct gender values ​​for all records in the training dataset. You need to send a file similar to `targets.tsv` from the training set to the system. That is, for each `id.wav` file in the test dataset, the response file should contain a line like `id\tgender `

## Model evaluation

The model I used is a sequential neural network that consists of an input layer with 512 neurons, 15 hidden layers (7 fully connected layers with 512, 256, 256, 128, 128, 64, 64 neurons in each, 8 dropout layers with a frequencies of 0.2-0.3), and an output layer with one neuron and a sigmoid activation function.

I chose an accuracy as the metric, since the dataset is balanced, and the Adam optimizer. After training with 130 epochs the model showed the accuracy of 98.01%.

## Technologies

  - Python 3.8
  - TensorFlow 2.x.x
  - Librosa
  - Numpy
  - Pandas

## Project structure

    .
    ├── __pycache__          # Cached files
    ├── logs                 # Logs of the tensorboard
    ├── results              # The feature matrix, the label vector and the NN's weights
    ├── test                 # The extracted features of the test dataset
    ├── train                # The extracted features of the training dataset
    ├── answers.tsv          # The answers for the test dataset
    ├── preprocessing.py     # Extracts features from the raw audio files
    ├── test.py              # Predicts the gender for the test dataset
    ├── train.py             # Trains the NN on the training dataset
    ├── utils.py             # Loads the data, splits it into train/test, creates the model
    └── README.md
