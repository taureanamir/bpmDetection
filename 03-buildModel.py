import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, splitext

TRAIN_DIR = 'data/figures/data/'
LABEL_DIR = 'data/annotations/tempo/'
TEST_DIR = 'data/figures/test/'
IMG_SIZE = 256
LR = 1e-3

MODEL_NAME = 'bpm_detection-{}-{}.model'.format(LR, '5conv-BPM-10KFoldDropout0.2InClassRun') # just so we remember which saved model is which, sizes must match
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
    if label < 70:
        return [1, 0, 0, 0, 0]
    elif label >= 70 and label < 100:
        return [0, 1, 0, 0, 0]
    elif label >= 100 and label < 135:
        return [0, 0, 1, 0, 0]
    elif label >= 135 and label < 160:
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
#data = create_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, splitext
import tensorflow as tf
from sklearn.model_selection import KFold

tf.reset_default_graph()

LOG_DIR = 'logs/'

LR = 1e-3
MODEL_NAME = 'bpm_detection-{}-{}.model'.format(LR, '5conv-BPM-10KFoldDropout0.2InClassRun') # just so we remember which saved model is which, sizes must match
IMG_SIZE = 256
NUM_CLASSES = 5
kf = KFold(n_splits=10, shuffle=True, random_state=1)

data = np.load('data.npy')
kf.get_n_splits(data)

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.2)

convnet = fully_connected(convnet, NUM_CLASSES, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=3)

# k-folds (10)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = data[train_index], data[test_index]

    X = np.array([i[0] for i in X_train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in Y_train]

    test_x = np.array([i[0] for i in X_test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in Y_test]

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
