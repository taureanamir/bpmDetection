import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, splitext

TRAIN_DIR = 'data/figures/data/'
LABEL_DIR = 'data/annotations/tempo/'
#TEST_DIR = 'data/figures/test/'
IMG_SIZE = 256
LR = 1e-3

#MODEL_NAME = 'bpm_detection-{}-{}.model'.format(LR, '2conv-BPM') # just so we remember which saved model is which, sizes must match
ext = '.bpm'

file_bpm_dict = {} # Create an empty dict

# List of training files
trainingDataFiles = [f for f in listdir(LABEL_DIR) if isfile(join(LABEL_DIR,f))]

# List of BPM files
bpm_files = [i for i in trainingDataFiles if splitext(i)[1] == ext]

# Iterate over bpm files
for f in bpm_files:
    # Open them and assign them to file_dict
    with open(os.path.join(LABEL_DIR,f)) as file_object:
        file_bpm_dict[f] = file_object.read()

# Iterate over your dict and print the key/val pairs.
#for i in file_bpm_dict:
#    print (i, file_bpm_dict[i])

#del trainingDataFiles
#del LABEL_DIR

def retrieve_label(filename):
    filename = filename + ".LOFI.bpm"
    label = file_bpm_dict[filename] # to return the BPM from the file
    label = float(label)
    if label < 65:
        return [1, 0, 0, 0, 0]
    elif label >= 65 and label < 90:
        return [0, 1, 0, 0, 0]
    elif label >= 90 and label < 120:
        return [0, 0, 1, 0, 0]
    elif label >= 120 and label < 150:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


def create_data():
    data = []
    for image in tqdm(os.listdir(TRAIN_DIR)):
        filename = image.split('.')[-3] # to extract only the first part of the filename
        label = retrieve_label(filename)
        path = os.path.join(TRAIN_DIR,image)
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(image),np.array(label)])

    shuffle(data)
    np.save('data.npy', data)
    return data



# Create  data
data = create_data()
