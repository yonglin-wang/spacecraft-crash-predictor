# Date: 2021/1/30 11:01 PM

from typing import List

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

import consts as C


def build_keras_cnn(sampling_rate: int, feature_num: int, using_seq_label: bool,
                    layer_sizes: List[int]=None,
                    filter_number: int=128,
                    dropout_rate: float=0.5,
                    threshold: float=0.5
                    ) -> Sequential:
    """create a Keras CNN model. Code based on https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/?fbclid=IwAR2VpQ2g-i5i-7Grgqj5JjtDXYrHBmOJcHeiyofyjOpM3RQFmxqYz1TvD1Q"""
    if using_seq_label:
        raise NotImplementedError("sequence label not supported in CNN!")
    if layer_sizes is None:
        layer_sizes = C.DEFAULT_CNN_LAYERS
    assert len(layer_sizes) > 1, "CNN must specify at least 1 layer number"

    # CNN
    model = Sequential()
    model.add(Conv1D(filters=filter_number, kernel_size=layer_sizes[0], activation='relu', input_shape=(sampling_rate, feature_num), padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    # use padding=same to avoid negative dimension issues
    for kernel_size in layer_sizes[1:]:
        model.add(Conv1D(filters=filter_number, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(Dropout(dropout_rate))
        model.add(MaxPooling1D(pool_size=2, padding='same'))

    # cap with FFNN
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=C.build_metrics_list(threshold)
    )
    return model


def build_keras_mlp(sampling_rate: int, feature_num: int, using_seq_label: bool,
                    layer_sizes: List[int]=None,
                    threshold=0.5
                    ) -> Sequential:
    """

    :param sampling_rate:
    :param feature_num:
    :param using_seq_label:
    :param layer_sizes: list of hidden layer size, not including first layer and last
    :param threshold:
    :return:
    """
    if using_seq_label:
        raise NotImplementedError("sequence label not supported in MLP!")
    if layer_sizes is None:
        layer_sizes = C.DEFAULT_MLP_HIDDEN

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=sampling_rate * feature_num, activation='relu'))
    for hidden_size in layer_sizes:
        model.add(Dense(units=hidden_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=C.build_metrics_list(threshold)
    )
    return model


if __name__ == "__main__":
    pass
