# Author: Yonglin Wang
# Date: 12/3/21 10:39 AM

from tensorflow.keras import Sequential

import consts as C

def compile_model(model: Sequential, threshold: float,
                  loss='binary_crossentropy',
                  optimizer='adam') -> Sequential:
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=C.build_metrics_list(threshold)
    )
    return model


if __name__ == "__main__":
    pass
