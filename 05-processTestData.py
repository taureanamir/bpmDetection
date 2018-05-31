import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, splitext

TEST_DIR = 'data/figures/test/'
IMG_SIZE = 256
LR = 1e-3

MODEL_NAME = 'bpm_detection-{}-{}.model'.format(LR, '5conv-BPM') # just so we remember which saved model is which, sizes must match

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# Create  data
data = process_test_data()

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
NUM_CLASSES = 5

LOG_DIR = 'logs/'

data = np.load('test_data.npy')

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

convnet = fully_connected(convnet, NUM_CLASSES, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

img_data = data[0]

data = img_data[0].reshape(IMG_SIZE,IMG_SIZE,1)

model_out = model.predict([data])[0]

print (model_out)

print ("Predicted Tempo")
print ("-----------------------------------------------------------------------")
if np.argmax(model_out) == 0:
    str_label='Very Slow (BPM 70 or less)'
elif np.argmax(model_out) == 1:
    str_label='Slow (BPM between 70 and 100)'
elif np.argmax(model_out) == 2:
    str_label='Medium (BPM between 100 and 135)'
elif np.argmax(model_out) == 3:
    str_label='High (BPM between 135 and 160)'
else:
    str_label='Very High (BPM 160 or above)'
print (str_label)
print ("-----------------------------------------------------------------------")
