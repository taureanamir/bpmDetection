# -*- coding: utf-8 -*-
# give argument for batch while executing this script
# e.g. python DataPreperation.py 1
"""
Data Preperation.
"""
import multiprocessing
import sys

from os import listdir, makedirs
from os.path import isfile, join, splitext

# Make a list of all files in our traning audio directory.

batch_num = sys.argv[1]
print ("Processing started for batch" + batch_num)

batch_dir = "batch" + batch_num

dataPath = "data/wav/" + batch_dir
trainingDataFiles = [f for f in listdir(dataPath) if isfile(join(dataPath,f))]

# Filter out the files and only include .wav files for analysis
trainingDataFiles = [x for x in trainingDataFiles if splitext(x)[1] == ".wav"]

# Make a mapping between the training data name and the path to the audio clip.
trainingData = {};
for data in trainingDataFiles:
    trainingData[splitext(data)[0]] = join(dataPath, data)


del trainingDataFiles
del dataPath

figurePath = "data/figures/" + batch_dir
makedirs(figurePath, exist_ok=True)

def generatePrecussionMap(name):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    y, sr = librosa.load(trainingData[name])
    D = librosa.stft(y)
    rp = np.max(np.abs(D))
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin = 8)
    librosa.display.specshow(librosa.amplitude_to_db(librosa.magphase(D_percussive)[0], ref=rp), y_axis='log')
    plt.axis('off')
    plt.savefig(join(figurePath, name) + ".png", bbox_inches='tight')
    print('.', end='')


# Try to parallize the above function
# Using process pool to avoid global interperter lock

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

    print ("Total files: %d" % (len(trainingData)))
    print ("Each dot represents a processed file")

    pool.map(generatePrecussionMap, trainingData.keys())
    pool.join()
    pool.close()
