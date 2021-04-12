# Author: Yonglin Wang
# Date: 2021/2/13 4:48 PM
"""Generate split indices for dataset either via Stratified K-Fold or leaving out N participants"""

from typing import Union, List, Tuple

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np

import consts as C

class Splitter:
    def __init__(self,
                 mode: str,
                 n_splits: int,
                 verbose=True):
        # check preconditions
        if mode not in C.SPLIT_MODES:
            raise ValueError(f"Cannot recognize mode name {mode}")
        if n_splits < 1:
            raise ValueError(f"Number of splits ({n_splits}) must be positive.")

        self.n_splits = n_splits
        self.verbose = verbose
        self.mode = mode

        # create internal splitter for each mode
        if self.mode == C.NO_CV or n_splits == 1:
            # if no split, no splitter used
            self.splitter = None
        else:
            if self.mode == C.KFOLD:
                # use stratified K to split data
                self.splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=C.RANDOM_SEED)
            elif self.mode == C.LEAVE_OUT:
                # use K fold to split participant names
                self.splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=C.RANDOM_SEED)
            else:
                raise NotImplementedError(f"{self.mode} not implemented!")


    def split_ind_generator(self,
                            y: np.ndarray,
                            subject_names: Union[List, np.ndarray]=None):
        if self.n_splits == 1 or self.mode == C.NO_CV:
            return self.__no_iter(y)

        else:
            if self.mode == C.KFOLD:
                return self.__kfold(y)
            elif self.mode == C.LEAVE_OUT:
                if not subject_names:
                    raise ValueError(f"Subject names must be provided to perform leave-{self.n_splits}-out by subject!")
                return self.__leave_out(y, subject_names)

    def __no_iter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """placeholder iterator that yields one split"""
        train_inds, test_inds = train_test_split(np.arange(y.shape[0]),
                                                 test_size=C.VAL_SIZE,
                                                 random_state=C.RANDOM_SEED)
        yield train_inds, test_inds

    def __kfold(self, y: np.ndarray):
        """wrapper for sklearn stratified K fold iterator"""
        assert type(self.splitter) == StratifiedKFold
        split_gen = self.splitter.split(np.zeros(y.shape[0]), y)
        while 1:
            try:
                train_inds, test_inds = next(split_gen)

                # must shuffle order, or RNN won't learn!!
                np.random.shuffle(train_inds)
                np.random.shuffle(test_inds)

                yield train_inds, test_inds
            except StopIteration:
                return

    def __leave_out(self, y, names):
        """generate data based on subject names"""
        # TODO implement with try-except StopIteration, don't forget to shuffle output
        raise NotImplementedError()

if __name__ == "__main__":
    s = Splitter(C.KFOLD, 5)
    # X = np.random.random((10,2))
    y = np.random.randint(0, 1, (10, 1))
    names = np.array(list("aabbcccddd"))

    for n in enumerate(s.split_ind_generator(y)):
        print(n)