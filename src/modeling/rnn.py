# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/30 10:38 PM
"""functions to return a built RNN model. Classes will be used if complicated deep learning architecture is involved."""


import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras import Sequential

import consts as C


def build_keras_rnn(sampling_rate, feature_num,
                    rnn_out_dim=128,
                    dropout_rate=0.5,
                    rnn_type=C.LSTM,
                    threshold=0.5
                    ) -> Sequential:
    """"""
    model_out = Sequential()

    if rnn_type == C.LSTM:
        model_out.add(
            keras.layers.LSTM(
                units=rnn_out_dim,
                input_shape=[sampling_rate, feature_num]
            )
        )
    elif rnn_type == C.GRU:
        model_out.add(
            keras.layers.GRU(
                units=rnn_out_dim,
                input_shape=[sampling_rate, feature_num]
            )
        )
    else:
        raise NotImplementedError("RNN does not recognize {}".format(rnn_type))

    model_out.add(keras.layers.Dropout(rate=dropout_rate, seed=C.RANDOM_SEED))
    model_out.add(keras.layers.Dense(units=rnn_out_dim, activation='relu'))
    model_out.add(keras.layers.Dense(1, activation='sigmoid'))
    model_out.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(name=C.ACC, threshold=threshold),
                 tf.keras.metrics.AUC(name=C.AUC),
                 tf.keras.metrics.Precision(name=C.PRECISION),
                 tf.keras.metrics.Recall(name=C.RECALL)]
        # metrics=[tf.keras.metrics.AUC()]
    )
    return model_out


if __name__ == "__main__":
    model = build_keras_rnn(50, 3)
