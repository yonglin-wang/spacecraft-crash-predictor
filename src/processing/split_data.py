# Author: Yonglin Wang
# Date: 2021/2/13 4:48 PM
"""Generate split indices for dataset either via Stratified K-Fold or leaving out N participants"""

from typing import Union, List, Tuple, Dict
import os

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np

import consts as C
from processing.episode_raw_readings import get_crash_non_crash_ids


class EpisodeKFoldSplitter:
    """Class for K-fold CV with splitting by episode; API similar to sklearn.model_selection.StratifiedKFold"""
    def __init__(self, n_splits: int, episode_ids: np.ndarray, shuffle=True, random_state=C.RANDOM_SEED):
        self.n_splits = n_splits
        self.episode_ids = episode_ids

        # get crash and noncrash episode ids based on episode_ids
        unique_episode_ids = set(self.episode_ids)
        crash_ids, non_ids = get_crash_non_crash_ids()
        self.crash_epi_ids = np.array(list(set(crash_ids).intersection(unique_episode_ids)))
        self.non_crash_epi_ids = np.array(list(set(non_ids).intersection(unique_episode_ids)))
        assert len(self.crash_epi_ids) != 0, "Empty crash episode ID for splitting!"
        assert len(self.non_crash_epi_ids) != 0, "Empty noncrash episode ID for splitting!"

        self.kf = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """return window indices based on splitting by episode; y only used for initial length checking"""
        assert len(y) == len(self.episode_ids), "episode ID length should be the same as label length！"

        # these split unique episode ids
        crash_epi_gen = self.kf.split(self.crash_epi_ids)
        non_crash_epi_gen = self.kf.split(self.non_crash_epi_ids)

        for (crash_train_inds, crash_test_inds), (noncrash_train_inds, noncrash_test_inds) \
            in zip(crash_epi_gen, non_crash_epi_gen):
            # use split indices to first infer train-test episode split
            train_episode_id = self.crash_epi_ids[crash_train_inds].tolist() + self.non_crash_epi_ids[noncrash_train_inds].tolist()
            test_episode_id = self.crash_epi_ids[crash_test_inds].tolist() + self.non_crash_epi_ids[noncrash_test_inds].tolist()

            train_inds = np.where(np.isin(self.episode_ids, train_episode_id))[0]
            test_inds = np.where(np.isin(self.episode_ids, test_episode_id))[0]

            # must shuffle order, or RNN won't learn!!
            np.random.shuffle(train_inds)
            np.random.shuffle(test_inds)

            yield train_inds, test_inds


class Splitter:
    def __init__(self,
                 mode: str,
                 n_splits: int,
                 episode_ids: np.ndarray,
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
                # use customized episode splitter to split data
                self.splitter = EpisodeKFoldSplitter(self.n_splits, episode_ids,
                                                     shuffle=True, random_state=C.RANDOM_SEED)
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
        """wrapper for EpisodeKFoldSplitter iterator"""
        assert type(self.splitter) == EpisodeKFoldSplitter

        return self.splitter.split(y)

    def __leave_out(self, y, names):
        """generate data based on subject names"""
        # TODO implement with try-except StopIteration, don't forget to shuffle output
        raise NotImplementedError()


def _save_test_train_split(episode_ids: list,
                           out_dir:str,
                           valid_non_crashed_ids: Dict[str, List[int]],
                           valid_crashed_ids: Dict[str, List[int]]):
    """save stratified test vs. train+val splits"""
    # change to ndarray for faster indexing
    episode_ids = np.array(episode_ids)

    # convert {<trial key>: [<episode id>]} dict to lists of ints
    crash_id_list = [id for id_list in valid_crashed_ids.values() for id in id_list]
    non_crash_id_list = [id for id_list in valid_non_crashed_ids.values() for id in id_list]

    # split episode ids in each class and combine
    crash_train, crash_test = train_test_split(crash_id_list, test_size=0.1, random_state=C.RANDOM_SEED, shuffle=True)
    non_crash_train, non_crash_test = train_test_split(non_crash_id_list, test_size=0.1, random_state=C.RANDOM_SEED, shuffle=True)
    train_episode_id = crash_train + non_crash_train
    test_episode_id = crash_test + non_crash_test

    # extract corresponding input train and test indices based on input episode ids
    train_inds = np.where(np.isin(episode_ids, train_episode_id))[0]
    test_inds = np.where(np.isin(episode_ids, test_episode_id))[0]

    # save split indices
    np.save(os.path.join(out_dir, C.INDS_PATH['train']), np.array(train_inds))
    np.save(os.path.join(out_dir, C.INDS_PATH['test']), np.array(test_inds))

    return train_inds, test_inds


if __name__ == "__main__":
    s = Splitter(C.KFOLD, 5, np.load("/Users/Violin/GitHub/spacecraft-crash-predictor/data/2000window_1000ahead_100rolling/episode_id.npy"))
    # X = np.random.random((10,2))
    y = np.random.randint(0, 1, (10, 1))
    names = np.array(list("aabbcccddd"))

    for n in enumerate(s.split_ind_generator(y)):
        print(n)