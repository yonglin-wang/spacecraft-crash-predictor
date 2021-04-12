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
                rnn_out_dim,
                input_shape=[sampling_rate, feature_num]
            )
        )
    elif rnn_type == C.GRU:
        model_out.add(
            keras.layers.GRU(
                rnn_out_dim,
                input_shape=[sampling_rate, feature_num]
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
        model_out.add(keras.layers.LSTM(rnn_out_dim))
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
    )
    return model_out


# build many to many RNN with timedistributed: https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/


if __name__ == "__main__":
    model = Sequential()
    model.add(keras.layers.LSTM(5, input_shape=(50,3), return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    inputs = tf.random.normal([32,50,3])
    output = model(inputs)
    print(model.summary())
    print(output.shape)
