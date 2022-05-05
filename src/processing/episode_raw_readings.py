#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/3/14 9:48 PM
import pickle
from collections import OrderedDict, defaultdict
import os
from typing import Tuple, List

import pandas as pd
import numpy as np
from tqdm import tqdm

import consts as C


def extract_segments(key_entries, current_seg_id: int):
    """extracts segment dictionary from trial entries, returns both dictionary and int ID for first seg ID for next trial"""
    # keep only copy to avoid making change to original data
    key_entries = key_entries.copy()
    # # group segments by temporary IDs
    key_entries["humanTimeID"] = (key_entries.trialPhase != key_entries.trialPhase.shift()).cumsum()

    # create info
    segment_info = OrderedDict()
    for i, seg in key_entries.groupby("humanTimeID"):
        # add basic info to segment dict
        seg_dict = {}
        seg_dict["duration"] = seg.seconds.iloc[-1] - seg.seconds.iloc[0]
        seg_dict["reading_num"] = len(seg.index)
        seg_dict["phase"] = seg.trialPhase.iloc[0]
        seg_dict["indices"] = seg.index.tolist()
        seg_dict["trial_key"] = seg.peopleTrialKey.iloc[0]
        seg_dict["subject"] = seg.peopleName.iloc[0]
        seg_dict["start_sec"] = seg.seconds.iloc[0]
        seg_dict["end_sec"] = seg.seconds.iloc[-1]

        # crashed: if current phase is 3, next index within this trial, next index is 4
        crash_ind = -1  # default to -1
        if seg_dict["phase"] == 3:
            next_ind = seg.index[-1] + 1
            if next_ind in key_entries.index:
                if key_entries.trialPhase.loc[next_ind] == 4:
                    crash_ind = next_ind
        seg_dict["crash_ind"] = crash_ind

        # add to overall dictionary
        segment_info[current_seg_id] = seg_dict
        current_seg_id += 1

    return segment_info, current_seg_id


def display_all_segments(seg_dict, output_cols) -> pd.DataFrame:
    record_list = []
    for seg_id, d in seg_dict.items():
        rec = {C.ID: seg_id}
        rec.update({k: d[k] for k in output_cols})
        record_list.append(rec)
    return pd.DataFrame.from_records(record_list)


def load_episode_data(raw_data_path: str, output_pickle_path: str, output_csv_path: str,
                      clean_data=True, verbose=True
                      ) -> Tuple[pd.DataFrame, OrderedDict]:
    """Load readings from raw data path, generate and save pickled segment information dictionary and segment stats DataFrame"""

    if not os.path.exists(output_pickle_path) or not os.path.exists(output_csv_path):
        df = pd.read_csv(raw_data_path, usecols=C.SEGMENT_COLS)

        all_segs = OrderedDict()
        start_id = 0

        with tqdm(desc="Generating segments from trials", total=df.peopleTrialKey.nunique(), disable=not verbose) as pbar:
            for _, trial_entries in df.groupby("peopleTrialKey"):
                # extract and save segments; update next available index
                seg_info, start_id = extract_segments(trial_entries, start_id)
                all_segs.update(seg_info)

                pbar.update(1)

        # Get output .csv table, validate, and save
        out_df = display_all_segments(all_segs, C.OUTPUT_COLS)
        assert out_df[C.ID].tolist() == out_df.index.tolist(), "index and id should match"
        out_df = out_df.set_index(C.ID)
        out_df.to_csv(output_csv_path)

        # save all_segs dictionary
        pickle.dump(all_segs, open(output_pickle_path, "wb"))

    else:
        out_df = pd.read_csv(output_csv_path, index_col="id")
        all_segs = pickle.load(open(output_pickle_path, "rb"))

    # clean metadata df, outputs only non-bug human episodes for further selection
    if clean_data:
        out_df = _clean_metadata(out_df)

    return out_df, all_segs


def _clean_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """remove buggy data entry from the given episode metadata df, return the debugged df"""
    # retain only human control episodes (phase = 3)
    meta_df = meta_df[meta_df.phase == 3]
    # locate buggy indices. current condition: single-reading human episodes and their corresponding crashes
    buggy_human_segs = meta_df[meta_df.reading_num <= C.MIN_ENTRIES_IN_WINDOW].index

    # different datasets have different numbers of buggy datasets
    print(f"excluding {len(buggy_human_segs)} buggy segments (i.e. segments with no more than {C.MIN_ENTRIES_IN_WINDOW} readings)")
    return meta_df.drop(meta_df[meta_df.index.isin(set(buggy_human_segs))].index)


def get_crash_non_crash_ids() -> Tuple[List[int], List[int]]:
    """return list of all crash episode ids and list of non-crash episode ids"""
    meta_df, episode_dict = load_episode_data(C.RAW_DATA_PATH, C.SEGMENT_DICT_PATH, C.SEGMENT_STATS_PATH,
                                              clean_data=True, verbose=False)
    crash_ids = meta_df[meta_df.crash_ind != -1].index.tolist()
    non_crash_ids = meta_df[meta_df.crash_ind == -1].index.tolist()

    return crash_ids, non_crash_ids


def count_episode_exclusion(time_gap: float):
    """test function for displaying exclusion statistics"""
    print(f"excluding episodes <= {time_gap} seconds")
    meta_df, segs_dict = load_episode_data(C.RAW_DATA_PATH, C.SEGMENT_DICT_PATH, C.SEGMENT_STATS_PATH)

    human_epi_count = {"orig_non": sum(meta_df.crash_ind == -1),
                       "orig_crash": sum(meta_df.crash_ind != -1)}
    # keep only episode_duration > gap
    meta_df = meta_df[meta_df.duration > time_gap]
    human_epi_count.update({"new_non": sum(meta_df.crash_ind == -1),
                            "new_crash": sum(meta_df.crash_ind != -1)})
    non_crash_excluded = human_epi_count['orig_non'] - human_epi_count['new_non']
    crash_excluded = human_epi_count['orig_crash'] - human_epi_count['new_crash']
    print(f"Human control episodes:\n"
                f"{human_epi_count['new_non']} out of {human_epi_count['orig_non']} non-crashed human control episodes "
                f"selected, {non_crash_excluded} ({non_crash_excluded/human_epi_count['orig_non']}) excluded\n"
                f"{human_epi_count['new_crash']} out of {human_epi_count['orig_crash']} crashed human control episodes "
                f"selected, {crash_excluded} ({crash_excluded/human_epi_count['orig_crash']}) excluded\n")

if __name__ == "__main__":
    # count_episode_exclusion(1)
    # count_episode_exclusion(2)
    # count_episode_exclusion(3)
    # count_episode_exclusion(4)
    # count_episode_exclusion(5)
    # count_episode_exclusion(6)
    count_episode_exclusion(2.3)
    count_episode_exclusion(2.5)
    count_episode_exclusion(2.7)