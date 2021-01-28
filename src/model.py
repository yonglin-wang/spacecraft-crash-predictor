import os
import glob
import pickle

import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
import time
# import matplotlib.pyplot as plt
# import warnings

import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# assertion packages
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend

from utils import display_exec_time
import argparse

# Control for random states
import random

SEED = 2020
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)  # `python` built-in pseudo-random generator
np.random.seed(SEED)  # numpy pseudo-random generator
tf.set_random_seed(SEED)  # tensorflow pseudo-random generator

# # set session after random seeds according to https://github.com/tensorflow/tensorflow/issues/18323
# sess = tf.Session(graph=tf.get_default_graph, config=tf.ConfigProto())
# keras.backend.set_session(sess)

# define format for saving tests TODO add destabilizing deflection file name
TRAIN_PATH_X = "data/data_{}ms/{}scale_{}ahead_train_X{}.npy"
TRAIN_PATH_Y = "data/data_{}ms/{}scale_{}ahead_train_y{}.npy"
TEST_PATH_X = "data/data_{}ms/{}scale_{}ahead_test_X{}.npy"
TEST_PATH_Y = "data/data_{}ms/{}scale_{}ahead_test_y{}.npy"

CALCULATED = "_calc"
ORIGINAL = "_orig"


# Argparser
# noinspection PyTypeChecker
argparser = argparse.ArgumentParser(prog="LSTM Trainer Argparser",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument(
    '--scale', type=float, default=1.0, help='time scale of training data, in seconds')
argparser.add_argument(
    '--ahead', type=float, default=0.5, help='prediction timing ahead of event, in seconds')
argparser.add_argument(
    '--cal_vel', action='store_true', help='whether to use calculated velocity instead of original')
argparser.add_argument(
    '--early_stop', action='store_true', help='whether to stop early training when converged')
argparser.add_argument(
    '--patience', type=int, default=3, help='max number of epochs allowed with no improvement, if early stopping')
argparser.add_argument(
    '--conv_crit', type=str.lower, default='loss', choices=['loss'],
    help='type of convergence criteria')    # TODO explore more crit

args = argparser.parse_args()

# convert seconds to ms
scale_ms = int(args.scale * 1000)
ahead_ms = int(args.ahead * 1000)
use_calculated = args.cal_vel
early_stopping = args.early_stop
patience_epochs = args.patience
conv_criteria = args.conv_crit

# print training info
print("Training information:")
print(f"Now training model with {scale_ms}ms scale, {ahead_ms}ms ahead.\n"
      f"Using calculated weight? {use_calculated}\n"
      f"Early Stopping? {early_stopping}\n")

if early_stopping:
    print(f"Stopping early if no {conv_criteria} improvement in {patience_epochs} epochs.\n")

if use_calculated:
    print("Using calculated velocity...")


#### Begin script
# confirm TensorFlow sees the GPU
# assert 'GPU' in str(device_lib.list_local_devices()), "TensorFlow cannot find GPU"
#
# # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
# assert len(tensorflow_backend._get_available_gpus()) > 0, "Keras cannot find GPU"

### time the script
begin = time.time()

def load_sets_after_split(ahead_ms, scale_ms, mode=ORIGINAL):
    if mode not in (CALCULATED, ORIGINAL):
        raise ValueError("Unknown velocity mode: {}".format(mode))

    return map(lambda format_path: np.load(open(format_path.format(scale_ms, ahead_ms, scale_ms, mode), "rb")),
               (TRAIN_PATH_X, TEST_PATH_X, TRAIN_PATH_Y, TEST_PATH_Y))

def save_sets_after_split(X_train, X_test, y_train, y_test, ahead_ms, scale_ms, mode=ORIGINAL):
    if mode not in (CALCULATED, ORIGINAL):
        raise ValueError("Unknown velocity mode: {}".format(mode))
    for format_path, arr in ((TRAIN_PATH_X, X_train), (TEST_PATH_X, X_test), (TRAIN_PATH_Y, y_train), (TEST_PATH_Y, y_test)):
        path = format_path.format(scale_ms, ahead_ms, scale_ms, mode)
        np.save(path, arr)
        print("array saved at {}".format(path))
    # map(lambda format_path, arr: np.save(open(format_path.format(scale_ms, ahead_ms, scale_ms, mode), "wb"), arr),  #TODO what's wrong??
    #     ((TRAIN_PATH_X, X_train), (TEST_PATH_X, X_test), (TRAIN_PATH_Y, y_train), (TEST_PATH_Y, y_test)))


# Now, save model and training stats
# ensure output folder exists
if not os.path.exists(f"results/results_{scale_ms}"):
    os.makedirs(f'results/results_{scale_ms}')

try:
    if use_calculated:
        X_train, X_test, y_train, y_test = load_sets_after_split(ahead_ms, scale_ms, mode=CALCULATED)
    else:
        X_train, X_test, y_train, y_test = load_sets_after_split(ahead_ms, scale_ms)

except:
    print("did not find train and test files... now generating...")
    #### read all the positive samples TODO merge these paths with generate data (use 1 var and avoid hard coding)
    crash_featurized = pd.read_pickle(f'data/data_{scale_ms}ms/crash_feature_label_{ahead_ms}ahead_{scale_ms}scale_test')

    ### read all the negative samples
    noncrash_featurized = pd.read_pickle(f'data/data_{scale_ms}ms/noncrash_feature_label_{ahead_ms}ahead_{scale_ms}scale_test')

    #### merge both positive and negative together
    data_final = pd.concat([crash_featurized, noncrash_featurized])
    data_final = data_final[['features_cal_vel','features_org_vel','label']]


    #### split the data with calculated velocity and original velocity seperately

    ###  calculated velocity
    # X_cal = data_final.features_cal_vel
    # X_cal = np.array([np.vstack(i) for i in X_cal])


    ### original velocity
    # choose velocity TODO change this to before loading data
    if use_calculated:
        X_all = np.array(data_final.features_cal_vel.to_list())
    else:
        X_all = np.array(data_final.features_org_vel.to_list())

    # X_all = np.array([np.vstack(i) for i in X_all])
    print("X_all shape: {}".format(X_all.shape))

    # y_all = np.array(data_final.label)

    y_all = data_final.label.to_numpy().reshape(-1, 1)
    print("y_all shape: {}".format(y_all.shape))
    print(f"Total crash instances: {np.count_nonzero(y_all == 1)}")
    # X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_cal, y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)

    if use_calculated:
        save_sets_after_split(X_train, X_test, y_train, y_test, ahead_ms, scale_ms, mode=CALCULATED)
    else:
        save_sets_after_split(X_train, X_test, y_train, y_test, ahead_ms, scale_ms)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
##### make data into sequence for training

###  calculated velocity
# X_train_cal = sequence.pad_sequences(X_train_cal, maxlen=50, padding='post', dtype='float', truncating='post')
# y_train_cal = np.array(y_train_cal).reshape(len(y_train_cal),1)

# X_test_cal = sequence.pad_sequences(X_test_cal, maxlen=50, padding='post', dtype='float', truncating='post')
# y_test_cal = np.array(y_test_cal).reshape(len(y_test_cal),1)

print("Processing Data...")

# pad sequences TODO trains don't need to be aligned?
X_train = sequence.pad_sequences(X_train, maxlen=50, padding='post', dtype='float', truncating='post')
# y_train = y_train.reshape(-1, 1)

X_test = sequence.pad_sequences(X_test, maxlen=50, padding='post', dtype='float', truncating='post')
# y_test = y_test.reshape(-1, 1)


#### onehotecoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

print("After padding X and OH encoding y...")
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

### train model

class_weights = [{
    0:1,
    1:1
},
{
    0:1,
    1:10
},
{
    0:1,
    1:50
}]

### try different weights
for i in range(len(class_weights)):
    print(f'---------Now training model with class weight {class_weights[i][0]}to{class_weights[i][1]}-------------')
    # initialize new model for this setting
    model = keras.Sequential()
    model.add(
        keras.layers.LSTM(
            units=128,
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
    )

    model.add(keras.layers.Dropout(rate=0.5, seed=SEED))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )

    # early stopping to save up GPU, stops training when there's no loss reduction in 4 epochs
    if early_stopping:
        callback = tf.keras.callbacks.EarlyStopping(monitor=conv_criteria, patience=patience_epochs)
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            class_weight = class_weights[i],
            callbacks=[callback]
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            class_weight=class_weights[i]
        )

    model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)


    predictions_org = y_pred[:, 0]
    predictions_org[predictions_org>=0.5] = 1
    predictions_org[predictions_org<0.5] = 0

    testing_org = y_test[:, 0]

    ### confusion matrix
    cf_array = confusion_matrix(testing_org, predictions_org)

    ### save the results with accuracy, recall, precision
    df_resutls = pd.DataFrame(cf_array)
    ### accuracy
    df_resutls['accuracy'] = (df_resutls.iloc[0,0] + df_resutls.iloc[1,1])/np.sum(cf_array)
    ### recall
    df_resutls['recall'] = df_resutls.iloc[0,0]/(df_resutls.iloc[0,0] + df_resutls.iloc[0,1])
    ### precision
    df_resutls['precision'] = df_resutls.iloc[0,0]/(df_resutls.iloc[0,0] + df_resutls.iloc[1,0])


    # save model to path, can be loaded with, e.g., reconstructed_model = keras.models.load_model(path_to_folder)
    model.save(f'results/results_{scale_ms}/{scale_ms}scale_{ahead_ms}ahead_{class_weights[i][0]}to{class_weights[i][1]}_model_orig_vel')
    # save test set stats
    df_resutls.to_csv(f'results/results_{scale_ms}/{scale_ms}scale_{ahead_ms}ahead_{class_weights[i][0]}to{class_weights[i][1]}_stats_orig_vel.csv')
    # pickle model training history
    pickle.dump(history, open(f'results/results_{scale_ms}/{scale_ms}scale_{ahead_ms}ahead_{class_weights[i][0]}to{class_weights[i][1]}_history_orig_vel.pkl', "wb"))

display_exec_time(begin, scr_name="model.py")
    # print(predictions_cal.shape, testing_cal.shape)

