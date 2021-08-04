# Author: Yonglin Wang
# Date: 2021/1/30 10:38 PM
"""functions to return a built RNN model. Classes will be used if complicated deep learning architecture is involved."""


import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed

import consts as C


def build_keras_rnn(sampling_rate, feature_num, using_seq_label: bool,
                    rnn_out_dim=128,
                    dropout_rate=0.5,
                    rnn_type=C.LSTM,
                    threshold=0.5
                    ) -> Sequential:
    """"""
    model_out = Sequential()

    # add RNN
    if rnn_type == C.LSTM:
        model_out.add(
            keras.layers.LSTM(
                units=rnn_out_dim,
                input_shape=[sampling_rate, feature_num],
                return_sequences=using_seq_label
            )
        )
    elif rnn_type == C.GRU:
        model_out.add(
            keras.layers.GRU(
                units=rnn_out_dim,
                input_shape=[sampling_rate, feature_num],
                return_sequences=using_seq_label
            )
        )
    elif rnn_type == C.LSTM_LSTM:
        # tutorial found here: https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        model_out.add(
            keras.layers.LSTM(
                rnn_out_dim,
                return_sequences=True,
                input_shape=[sampling_rate, feature_num]
            )
        )
        model_out.add(keras.layers.LSTM(rnn_out_dim, return_sequences=using_seq_label))
    elif rnn_type == C.GRU_GRU:
        # tutorial found here: https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        model_out.add(
            keras.layers.GRU(
                rnn_out_dim,
                return_sequences=True,
                input_shape=[sampling_rate, feature_num]
            )
        )
        model_out.add(keras.layers.GRU(rnn_out_dim, return_sequences=using_seq_label))
    else:
        raise NotImplementedError("RNN does not recognize {}".format(rnn_type))

    # add FFNN
    if using_seq_label:
        # feed hidden state at each time step to the same FFNN
        # tutorial see: https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
        model_out.add(TimeDistributed(Dropout(rate=dropout_rate, seed=C.RANDOM_SEED)))
        model_out.add(TimeDistributed(Dense(units=rnn_out_dim, activation='relu')))
        model_out.add(TimeDistributed(Dense(1, activation='sigmoid')))
    else:
        model_out.add(Dropout(rate=dropout_rate, seed=C.RANDOM_SEED))
        model_out.add(Dense(units=rnn_out_dim, activation='relu'))
        model_out.add(Dense(1, activation='sigmoid'))

    # compile model
    model_out.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=C.build_metrics_list(threshold)
    )
    return model_out



if __name__ == "__main__":
    model = build_keras_rnn(50, 3, False, rnn_type=C.GRU_GRU)
    model.summary()

    model2 = build_keras_rnn(50, 3, False, rnn_type=C.GRU)
    model2.summary()
