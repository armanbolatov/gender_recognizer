import glob
import numpy as np
import csv
from tqdm import tqdm
from tensorflow.keras.models import load_model

model = load_model('results/model.h5')

with open('answers.tsv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    for file in tqdm(glob.glob("test/*.npy")):
        x = np.load(file).reshape(1, -1)
        male_prob = model.predict(x)[0][0]
        gender = 1 if male_prob > 0.5 else 0
        id = file[5:-4]
        writer.writerow([id, gender])
