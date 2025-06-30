from typing import List, Dict, Optional

import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from chorder import chord, dechorder
from pandarallel import pandarallel

from .custom_data.constants import MMT_MAPPING_PATH
from .utils import pickle_load

pandarallel.initialize(progress_bar=True)

SCALE = chord.Chord.default_scale
SCALE_ENHARMONIC = chord.Chord.note_conversion_table
NOTES_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


with open(MMT_MAPPING_PATH) as f:
    mmt_mapping = json.load(f)
instr2program = mmt_mapping["instrument_program_map"]
instr_list_default = list(instr2program.keys())

DEFAULT_INSTR = -1
DEFAULT_INSTR_IDX = -1


data_dir_default = "remi_dataset"
polyph_out_dir_default = "remi_dataset/attr_cls/polyph"
rhythm_out_dir_default = "remi_dataset/attr_cls/rhythm"
harmdiv_out_dir_default = "remi_dataset/attr_cls/harmdiv"
rhythmdiv_out_dir_default = "remi_dataset/attr_cls/rhythmdiv"
rhythm_local_out_dir_default = "remi_dataset/attr_cls/local_rhythm"
pitch_local_out_dir_default = "remi_dataset/attr_cls/local_pitch"
config_out_path_default = "remi_dataset/attr_cls"

# 8 classes
polyphonicity_bounds = [0.0, 2.25, 4.0, 5.5, 7.25, 9.375, 12.1875, 16.875]
rhym_intensity_bounds = [0.0, 0.1875, 0.3125, 0.375, 0.5, 0.55, 0.65, 0.875]

harm_diversity_bounds = [0.0, 1.375, 2.0, 2.5625, 3.0, 3.375, 3.875, 4.625]
rhythmic_diversity_bounds = [0.0, 0.375, 0.6875, 1.0, 1.3125, 1.75, 2.3125, 3.25]
# For pitchclass seems better (?)
# harm_diversity_bounds = [0.0, 1.125, 1.8125, 2.25, 2.625, 3.0, 3.3125, 3.875]
# rhythmic_diversity_bounds = [0.0, 0.25, 0.357, 0.5, 0.55, 0.67, 0.83, 1.0]

rhym_intensity_tracks_bounds = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.30, 0.375, 0.5]
pitchavg_tracks_bounds = list(np.arange(0, 128, 10, dtype=float))

side = "right"


# ====================== ATTRIBUTES CONTROL ============================


def compute_polyphonicity(events: List[Dict], n_bars: int) -> np.ndarray:
    """Compute the bar-wise polyphonicity values of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars

    Returns
    -------
    np.ndarray
        Bar-wise polyphonicity values
    """
    poly_record = np.zeros((n_bars * 16,))

    cur_bar, cur_pos = -1, -1
    for ev in events:
        if ev["name"] == "Bar":
            cur_bar += 1
        elif ev["name"] == "Beat":
            cur_pos = int(ev["value"])
        elif ev["name"] == "Note_Duration":
            duration = int(ev["value"]) // 120
            st = cur_bar * 16 + cur_pos
            poly_record[st : st + duration] += 1

    return poly_record


def get_onsets_timing(events: List[Dict], n_bars: int) -> np.ndarray:
    """Compute the bar-wise rhytmicity values of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars

    Returns
    -------
    np.ndarray
        Bar-wise rhythmicity values
    """
    onset_record = np.zeros((n_bars * 16,))

    cur_bar, cur_pos = -1, -1
    for ev in events:
        if ev["name"] == "Bar":
            cur_bar += 1
        elif ev["name"] == "Beat":
            cur_pos = int(ev["value"])
        elif ev["name"] == "Note_Pitch" or ev["name"] == "Note_PitchClass":
            rec_idx = cur_bar * 16 + cur_pos
            onset_record[rec_idx] = 1

    return onset_record


def compute_harmonic_diversity(events: List[Dict], n_bars: int) -> np.ndarray:
    """Compute the bar-wise harmonic diversity values of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars

    Returns
    -------
    np.ndarray
        Bar-wise harmonic diversity values
    """
    chromagram = np.zeros((n_bars * 16, 12))

    cur_bar, cur_pos, cur_pitch = -1, -1, -1
    for ev in events:
        if ev["name"] == "Bar":
            cur_bar += 1
        elif ev["name"] == "Beat":
            cur_pos = int(ev["value"])
        elif ev["name"] == "Note_Pitch":
            cur_pitch = int(ev["value"]) % 12
        elif ev["name"] == "Note_PitchClass":
            cur_pitch = NOTES_NAMES.index(ev["value"])
        elif ev["name"] == "Note_Duration":
            duration = int(ev["value"]) // 120
            st = cur_bar * 16 + cur_pos
            chromagram[st : st + duration, cur_pitch] = 1

    pitches_diversity = chromagram.sum(axis=-1)
    return pitches_diversity


def compute_rhythmic_diversity(
    events: List[Dict],
    n_bars: int,
    instr_list: List[str] = instr_list_default,
    DEFAULT_INSTR_IDX: int = DEFAULT_INSTR_IDX,
) -> np.ndarray:
    """Compute the bar-wise rhythmic diversity values of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars
    instr_list : List[str], optional
        List of instruments involved in total, by default instr_list_default
    DEFAULT_INSTR_IDX : int, optional
        Index of the default instrument, by default DEFAULT_INSTR_IDX

    Returns
    -------
    np.ndarray
        Bar-wise rhythmic diversity values
    """

    instr_onset_record = np.zeros((64, n_bars, 16))
    cur_bar, cur_pos, cur_instrument = -1, -1, DEFAULT_INSTR_IDX
    for ev in events:
        if ev["name"] == "Bar":
            cur_bar += 1
        elif ev["name"] == "Beat":
            cur_pos = int(ev["value"])
        elif ev["name"] == "Track":
            cur_instrument = instr_list.index(ev["value"])
        elif ev["name"] == "Note_Pitch" or ev["name"] == "Note_PitchClass":
            instr_onset_record[cur_instrument, cur_bar, cur_pos] = 1

    # unique number of homorythms
    rhythmic_div_per_bar = np.zeros((n_bars))
    for ii_bar in range(instr_onset_record.shape[1]):
        all_instr_onsets = instr_onset_record[:, ii_bar]
        unique_onsets, counts = np.unique(all_instr_onsets, return_counts=True, axis=0)

        # check if zero
        subarray_indexes = np.arange(len(unique_onsets))
        index_zero = np.where(unique_onsets.sum(axis=1) == 0)
        if len(index_zero[0] != 0):
            subarray_indexes = np.delete(subarray_indexes, index_zero[0][0])

        if len(subarray_indexes) > 0:
            rhythmic_div_per_bar[ii_bar] = (
                len(unique_onsets[subarray_indexes]) / counts[subarray_indexes].sum()
            )
        else:
            rhythmic_div_per_bar[ii_bar] = 0

    return rhythmic_div_per_bar


def compute_onsets_timing_trackwise(
    events: List[Dict],
    n_bars: int,
    instr_list: List[str] = instr_list_default,
    DEFAULT_INSTR_IDX: int = DEFAULT_INSTR_IDX,
) -> Dict:
    """Compute the bar-wise / track-wise pitch diversity values of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars
    instr_list : List[str], optional
        List of instruments involved in total, by default instr_list_default
    DEFAULT_INSTR_IDX : int, optional
        Index of the default instrument, by default DEFAULT_INSTR_IDX

    Returns
    -------
    Dict
        Bar-wise / track-wise pitch diversity values
    """
    instr_onset_record = np.zeros((64, n_bars, 16))
    cur_bar, cur_pos, cur_instrument = -1, -1, DEFAULT_INSTR_IDX
    for ev in events:
        if ev["name"] == "Bar":
            cur_bar += 1
        elif ev["name"] == "Beat":
            cur_pos = int(ev["value"])
        elif ev["name"] == "Track":
            cur_instrument = instr_list.index(ev["value"])
        elif ev["name"] == "Note_Pitch" or ev["name"] == "Note_PitchClass":
            instr_onset_record[cur_instrument, cur_bar, cur_pos] = 1

    by_bar_onsets = instr_onset_record.mean(axis=-1)

    onset_sum = instr_onset_record.sum(axis=-1).sum(axis=1)
    instruments_present = np.where(onset_sum != 0)[0]

    # return dict([(k, by_bar_onsets[k]) for k in instruments_present])
    return dict([(instr_list[k], by_bar_onsets[k]) for k in instruments_present])


def compute_avg_pitch(
    events: List[Dict],
    n_bars: int,
    instr_list: List[str] = instr_list_default,
    DEFAULT_INSTR: int = DEFAULT_INSTR,
) -> Dict[str, np.ndarray]:
    """Compute the bar-wise / track-wise average pitches of a sequence of events

    Parameters
    ----------
    events : List[Dict]
        List of events as dicts {'name', 'value'}
    n_bars : int
        Number of bars
    instr_list : List[str], optional
        List of instruments involved in total, by default instr_list_default
    DEFAULT_INSTR_IDX : int, optional
        Index of the default instrument, by default DEFAULT_INSTR_IDX

    Returns
    -------
    Dict
        Bar-wise / track-wise average pitches
    """
    instr_bar_pitch = np.zeros((64, n_bars))
    cur_bar, cur_instrument = -1, DEFAULT_INSTR
    cur_pitchclass_id = None
    dict_bar_notes = dict([(instr, []) for instr in instr_list])
    for ev in events:
        if ev["name"] == "Bar":
            for ii, (_, notes) in enumerate(dict_bar_notes.items()):
                if len(notes) != 0:
                    instr_bar_pitch[ii, cur_bar] = np.mean(notes)
            cur_bar += 1
            dict_bar_notes = dict([(instr, []) for instr in instr_list])
        elif ev["name"] == "Track":
            cur_instrument = ev["value"]
        # case absolute pitch encoding
        elif ev["name"] == "Note_Pitch":
            dict_bar_notes[cur_instrument].append(ev["value"])  # type: ignore
        # case pitchclass pitch encoding
        elif ev["name"] == "Note_PitchClass":
            cur_pitchclass_id = NOTES_NAMES.index(ev["value"])
        elif ev["name"] == "Note_Octave":
            dict_bar_notes[cur_instrument].append(  # type: ignore
                (ev["value"] + 1) * 12 + cur_pitchclass_id
            )

    onset_sum = instr_bar_pitch.sum(axis=-1)
    instruments_present = np.where(onset_sum != 0)[0]

    return dict([(instr_list[k], instr_bar_pitch[k]) for k in instruments_present])


# ==================================================================================


def proc_one(
    piece: str,
    data_dir: str,
    polyph_out_dir: Optional[str] = None,
    rhythm_out_dir: Optional[str] = None,
    harmdiv_out_dir: Optional[str] = None,
    rhythmdiv_out_dir: Optional[str] = None,
    rhythm_local_out_dir: Optional[str] = None,
    pitch_local_out_dir: Optional[str] = None,
    melodic_tokens_position: Optional[str] = None,
    instr_list: List[str] = instr_list_default,
) -> Dict:
    bar_pos, events = pickle_load(os.path.join(data_dir, piece))
    events = events[: bar_pos[-1]]

    # if melodic_tokens_position == 'before':
    #     pos_end = events.index({'name': 'MelodySeqEnd', 'value': None})
    #     events = events[pos_end+1:]

    # === MUSEMORPHOSE GLOBAL ATTRIBUTES ===
    # polyphonicity
    polyph_raw = np.reshape(
        compute_polyphonicity(events, n_bars=len(bar_pos)), (-1, 16)
    )
    polyph_cls = (
        np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1), side=side) - 1  # type: ignore
    ).tolist()
    polyph_val = np.mean(polyph_raw, axis=-1)

    # rhythmicity
    rhythm_raw = np.reshape(get_onsets_timing(events, n_bars=len(bar_pos)), (-1, 16))
    rfreq_cls = (
        np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1), side=side) - 1  # type: ignore
    ).tolist()
    rfreq_val = np.mean(rhythm_raw, axis=-1)

    # === MULTITRACK GLOBAL ATTRIBUTES ===
    # multitrack harmonic diversity
    harm_diversity_raw = np.reshape(
        compute_harmonic_diversity(events, n_bars=len(bar_pos)), (-1, 16)
    )
    harmdiv_cls = (
        np.searchsorted(
            harm_diversity_bounds, np.mean(harm_diversity_raw, axis=-1), side=side  # type: ignore
        )
        - 1
    ).tolist()
    harmdiv_val = np.mean(harm_diversity_raw, axis=-1)

    # rhythmic diversity
    rhythmdiv_val = compute_rhythmic_diversity(
        events, n_bars=len(bar_pos), instr_list=instr_list
    )
    rhythmdiv_cls = (
        np.searchsorted(rhythmic_diversity_bounds, rhythmdiv_val, side=side) - 1  # type: ignore
    ).tolist()

    # === MULTITRACK LOCAL ATTRIBUTES ===
    # multitrack rhythmicity
    dict_rfreq_multitrack_val = compute_onsets_timing_trackwise(events, len(bar_pos))
    dict_rfreq_multitrack_cls = dict(
        [(k, []) for k in dict_rfreq_multitrack_val.keys()]
    )
    for k, rfreq_track in dict_rfreq_multitrack_val.items():
        dict_rfreq_multitrack_cls[k] = (
            np.searchsorted(rhym_intensity_tracks_bounds, rfreq_track, side=side) - 1  # type: ignore
        ).tolist()

    # multitrack pitch avg
    dict_pitchavg_multitrack_val = compute_avg_pitch(events, n_bars=len(bar_pos))
    dict_pitchavg_multitrack_cls = dict(
        [(k, []) for k in dict_pitchavg_multitrack_val.keys()]
    )
    for k, pitchavg_track in dict_pitchavg_multitrack_val.items():
        dict_pitchavg_multitrack_cls[k] = (
            np.searchsorted(pitchavg_tracks_bounds, pitchavg_track, side=side) - 1  # type: ignore
        ).tolist()

    if polyph_out_dir is not None:
        pickle.dump(polyph_cls, open(os.path.join(polyph_out_dir, piece), "wb"))
    if rhythm_out_dir is not None:
        pickle.dump(rfreq_cls, open(os.path.join(rhythm_out_dir, piece), "wb"))
    if harmdiv_out_dir is not None:
        pickle.dump(harmdiv_cls, open(os.path.join(harmdiv_out_dir, piece), "wb"))
    if rhythmdiv_out_dir is not None:
        pickle.dump(rhythmdiv_cls, open(os.path.join(rhythmdiv_out_dir, piece), "wb"))
    if rhythm_local_out_dir is not None:
        pickle.dump(
            dict_rfreq_multitrack_cls,
            open(os.path.join(rhythm_local_out_dir, piece), "wb"),
        )
    if pitch_local_out_dir is not None:
        pickle.dump(
            dict_pitchavg_multitrack_cls,
            open(os.path.join(pitch_local_out_dir, piece), "wb"),
        )

    return {
        "polyph_val": polyph_val,
        "polyph_cls": polyph_cls,
        "rfreq_val": rfreq_val,
        "rfreq_cls": rfreq_cls,
        "harmdiv_val": harmdiv_val,
        "harmdiv_cls": harmdiv_cls,
        "rhythmdiv_val": rhythmdiv_val,
        "rhythmdiv_cls": rhythmdiv_cls,
        "local_rfreq_val": dict_rfreq_multitrack_val,
        "local_rfreq_cls": dict_rfreq_multitrack_cls,
        "local_pitchavg_val": dict_pitchavg_multitrack_val,
        "local_pitchavg_cls": dict_pitchavg_multitrack_cls,
    }


# ==================================================================================


def analyze_local_attributes(filename: str, data_dir: str) -> List[Dict]:
    """Analyze the track-wise attributes of a generated or analyzed event file

    Parameters
    ----------
    filename : str
        Filename of the events
    data_dir : str
        Path to the filename

    Returns
    -------
    List[Dict]
        Dict per bar of pitch-range and repeatability values
    """
    events = pickle_load(os.path.join(data_dir, filename))
    per_bar = []
    current_bar = {
        "pitchrange": [],
        "repeatability": [],
    }
    for ii, ev in enumerate(events):
        if ev["name"] == "Bar" and ii > 0:
            per_bar.append(current_bar)
            current_bar = {
                "pitchrange": [],
                "repeatability": [],
            }
        elif ev["name"] == "Description_PitchRange":
            split_data = ev["value"].split("-")
            current_bar["pitchrange"].append(
                ("-".join(split_data[:-1]), int(split_data[-1]))
            )
        elif ev["name"] == "Description_Repeatability":
            split_data = ev["value"].split("-")
            current_bar["repeatability"].append(
                ("-".join(split_data[:-1]), int(split_data[-1]))
            )

    return per_bar


# ==================================================================================


def main(
    data_dir: str = data_dir_default,
    polyph_out_dir: str = polyph_out_dir_default,
    rhythm_out_dir: str = rhythm_out_dir_default,
    harmdiv_out_dir: str = harmdiv_out_dir_default,
    rhythmdiv_out_dir: str = rhythmdiv_out_dir_default,
    rhythm_local_out_dir: str = rhythm_local_out_dir_default,
    pitch_local_out_dir: str = pitch_local_out_dir_default,
    config_out_path: str = config_out_path_default,
    melodic_tokens_position: Optional[str] = None,
    instr_list: List[str] = instr_list_default,
):

    print(f"[info] Using {len(polyphonicity_bounds)} classes")

    pieces = [p for p in sorted(os.listdir(data_dir)) if ".pkl" in p]

    if not os.path.exists(polyph_out_dir):
        os.makedirs(polyph_out_dir)
    if not os.path.exists(rhythm_out_dir):
        os.makedirs(rhythm_out_dir)
    if not os.path.exists(harmdiv_out_dir):
        os.makedirs(harmdiv_out_dir)
    if not os.path.exists(rhythmdiv_out_dir):
        os.makedirs(rhythmdiv_out_dir)
    if not os.path.exists(rhythm_local_out_dir):
        os.makedirs(rhythm_local_out_dir)
    if not os.path.exists(pitch_local_out_dir):
        os.makedirs(pitch_local_out_dir)

    df = pd.DataFrame({"pieces": pieces})

    def proc_one_path(piece):
        return proc_one(
            piece,
            data_dir,
            polyph_out_dir,
            rhythm_out_dir,
            harmdiv_out_dir,
            rhythmdiv_out_dir,
            rhythm_local_out_dir,
            pitch_local_out_dir,
            melodic_tokens_position=melodic_tokens_position,
            instr_list=instr_list,
        )

    df["features"] = df.pieces.parallel_apply(proc_one_path)
    # df['features'] = df.pieces.apply(proc_one_path)
    df["polyph"] = df["features"].apply(lambda x: x["polyph_val"])
    df["rfreq"] = df["features"].apply(lambda x: x["rfreq_val"])
    df["harmdiv"] = df["features"].apply(lambda x: x["harmdiv_val"])
    df["rhythmdiv"] = df["features"].apply(lambda x: x["rhythmdiv_val"])
    df["polyph_cls"] = df["features"].apply(lambda x: x["polyph_cls"])
    df["rfreq_cls"] = df["features"].apply(lambda x: x["rfreq_cls"])
    df["harmdiv_cls"] = df["features"].apply(lambda x: x["harmdiv_cls"])
    df["rhythmdiv_cls"] = df["features"].apply(lambda x: x["rhythmdiv_cls"])
    df["local_rfreq_cls"] = df["features"].apply(lambda x: x["local_rfreq_cls"])
    df["local_pitchavg_cls"] = df["features"].apply(lambda x: x["local_pitchavg_cls"])

    all_polyph_cls = np.concatenate(df.polyph_cls.tolist())
    all_rfreq_cls = np.concatenate(df.rfreq_cls.tolist())
    all_harmdiv_cls = np.concatenate(df.harmdiv_cls.tolist())
    all_rhythmdiv_cls = np.concatenate(df.rhythmdiv_cls.tolist())

    with open(os.path.join(config_out_path, "config_bounds.json"), "w") as f:
        json.dump(
            {
                "polyphonicity_bounds": polyphonicity_bounds,
                "rhym_intensity_bounds": rhym_intensity_bounds,
                "harm_diversity_bounds": harm_diversity_bounds,
                "rhythmic_diversity_bounds": rhythmic_diversity_bounds,
                "rhym_intensity_tracks_bounds": rhym_intensity_tracks_bounds,
                "pitchavg_tracks_bounds": pitchavg_tracks_bounds,
            },
            f,
            indent=2,
        )

    print()
    print("[polyph classes]", Counter(all_polyph_cls))
    print("[rhythm classes]", Counter(all_rfreq_cls))
    print("[harmdiv classes]", Counter(all_harmdiv_cls))
    print("[rhythmdiv classes]", Counter(all_rhythmdiv_cls))


if __name__ == "__main__":
    main()
