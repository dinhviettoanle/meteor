import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .evaluate_fun import (
    compute_by_beat_melodic_fidelity,
    compute_full_sequence_melodic_fidelity,
)
from .utils import pickle_load


# ============================================================================================
# ====================================== MELODIC FIDELITY ====================================
# ============================================================================================


def make_bar_beat_melody_dict(melody_tokens):
    bar_beat_melody = dict()
    for bar_idx, bar_melody_tokens in enumerate(melody_tokens):
        bar_beat_melody[bar_idx] = dict()
        current_beat = 0
        for beat_idx, beat_melody_token in enumerate(bar_melody_tokens):
            str_token = f"{beat_melody_token['name']}_{beat_melody_token['value']}"
            if beat_melody_token["name"] == "Beat":
                current_beat = beat_melody_token["value"]
                bar_beat_melody[bar_idx][current_beat] = []
            else:
                if str_token not in bar_beat_melody[bar_idx][current_beat]:
                    bar_beat_melody[bar_idx][current_beat].append(str_token)

    return bar_beat_melody


def make_data_melodic_fidelity_figaro(filename):
    with open(filename, "r") as f:
        data_tokens = json.load(f)[0]

    dict_by_track_seq = dict()
    # Get all tracks
    for token in data_tokens:
        if token.startswith("Instrument"):
            track_value = token.split("_")[-1]
            dict_by_track_seq[track_value] = []

    # Remove duplicate position
    data_tokens_no_duplicate_beat = []

    last_position = None
    for token in data_tokens:
        if token.startswith("Bar"):
            last_position = None
            data_tokens_no_duplicate_beat.append(token)
        elif token.startswith("Position"):
            if token != last_position:
                data_tokens_no_duplicate_beat.append(token)
            last_position = token
        else:
            data_tokens_no_duplicate_beat.append(token)

    data_tokens = data_tokens_no_duplicate_beat

    # Get track-wise token sequence
    current_track = None
    for token in data_tokens:
        if token.startswith("Instrument"):
            current_track = token.split("_")[-1]
        elif token.startswith("Bar") or token.startswith("Position"):
            for _, l in dict_by_track_seq.items():
                if len(l) == 0 or (len(l) > 0 and l[-1] != token):
                    l.append(token)
        elif token.startswith("Pitch") or token.startswith(
            "Duration"
        ):  # and not('Duration' in token)
            dict_by_track_seq[current_track].append(token)

    # Convert into monophonic tracks (skyline)
    monophonic_track_tokens = dict((trk, []) for trk in dict_by_track_seq.keys())
    for track, track_tokens in dict_by_track_seq.items():
        beat_content = []
        for ii, token in enumerate(track_tokens):
            if token.startswith("Position") or token.startswith("Bar"):
                if len(beat_content) <= 3:
                    monophonic_track_tokens[track] += beat_content
                else:
                    highest_pitch = sorted(
                        [x for x in beat_content if "Pitch" in x], reverse=True
                    )[0]
                    highest_pitch_index = beat_content.index(highest_pitch)
                    monophonic_track_tokens[track] += [
                        beat_content[0],
                        beat_content[highest_pitch_index],
                        beat_content[highest_pitch_index + 1],
                    ]
                beat_content = []
                if token.startswith("Bar"):
                    monophonic_track_tokens[track].append(token)

            if not (token.startswith("Bar")):
                beat_content.append(token)

        if len(beat_content) <= 3:
            monophonic_track_tokens[track] += beat_content
        else:
            highest_pitch = sorted(
                [x for x in beat_content if "Pitch" in x], reverse=True
            )[0]
            highest_pitch_index = beat_content.index(highest_pitch)
            monophonic_track_tokens[track] += [
                beat_content[0],
                beat_content[highest_pitch_index],
                beat_content[highest_pitch_index + 1],
            ]

    # Split by bar
    track_tokens_by_bar = dict((trk, []) for trk in dict_by_track_seq.keys())
    for track, track_tokens in monophonic_track_tokens.items():
        bar_content = []
        ii_bar = 0
        for token in track_tokens:
            if token.startswith("Bar") and len(bar_content) > 0:
                track_tokens_by_bar[track].append(bar_content.copy())
                bar_content = []
                ii_bar += 1
            elif not (token.startswith("Bar")):
                bar_content.append(token)

        track_tokens_by_bar[track].append(bar_content.copy())

    n_bars = min([len(x) for x in track_tokens_by_bar.values()])

    by_bar_track_tokens = []
    for ii in range(n_bars):
        single_bar_by_tracks = dict((trk, []) for trk in dict_by_track_seq.keys())
        for trk in single_bar_by_tracks.keys():
            single_bar_by_tracks[trk] = track_tokens_by_bar[trk][ii]
        by_bar_track_tokens.append(single_bar_by_tracks)

    ## MELODY GROUNTRUTH
    # Get the same number of bars
    mid_to_idx = json.load(open("custom_data/files_symphonynet_full.json"))
    idx = [(k, v) for k, v in mid_to_idx.items() if f"/{filename.stem}" in v][0]

    # melody_filename = os.path.join('custom_data', 'symphonynet_full_multitrack_melodyfirst', 'melody', f'{idx[0]}.pkl')
    melody_filename = os.path.join(
        "/mnt/nfs_share_magnet2/ldinhvie/musemorphose_data/symphonynet_full_multitrack_melfull/melody",
        f"{idx[0]}.pkl",
    )
    melody_events = pickle_load(melody_filename)["melody_events"]

    chunk_melody_events = []
    by_bar_melody_events = []
    mel_n_bars = 0
    for event in melody_events:
        if event["name"] == "Bar" and len(by_bar_melody_events) > 0:
            mel_n_bars += 1
            chunk_melody_events.append(by_bar_melody_events)
            by_bar_melody_events = []
            if mel_n_bars == n_bars:
                break
        elif event["name"] != "Bar":
            by_bar_melody_events.append(event)

    # Make melody tokens dict
    melody_tokens_dict = make_bar_beat_melody_dict(chunk_melody_events)

    # Full sequence
    melody_tokens = []
    by_bar_melody_tokens = []
    for bar, bar_content in melody_tokens_dict.items():
        melody_tokens.append("Bar_None")
        bar_tokens = []
        for beat, beat_content in bar_content.items():
            beat_content_no_vel = [
                x for x in beat_content if "Velocity" not in x
            ]  ## and 'Duration' not in x
            beat_beat_content_no_vel = [f"Beat_{beat}"] + beat_content_no_vel
            melody_tokens += beat_beat_content_no_vel
            bar_tokens += beat_beat_content_no_vel
        by_bar_melody_tokens.append(bar_tokens)

    return (
        monophonic_track_tokens,
        melody_tokens,
        by_bar_melody_tokens,
        by_bar_track_tokens,
    )


def convert_figaro_to_common(tokens):
    new_tokens = []
    for token in tokens:
        if token.startswith("Bar"):
            new_tokens.append("Bar_None")
        else:
            new_tokens.append(token)
    return new_tokens


def convert_musemorphose_to_common(tokens):
    new_tokens = []
    for token in tokens:
        if token.startswith("Note"):
            new_tokens.append(token.replace("Note_", ""))
        elif token.startswith("Beat"):
            new_value = int(token.split("_")[-1]) * 3
            new_tokens.append(f"Position_{new_value}")
        else:
            new_tokens.append(token)
    return new_tokens


def melodic_similarity_figaro(path):
    all_full_seq_fidelity = []
    all_per_beat_fidelity = []
    all_per_beat_intersect = []

    for ii, filename in enumerate(Path(path).glob("*.json")):
        if "orig" in filename.name:
            continue
        # print(filename)
        print(ii, end=" ")
        (
            monophonic_track_tokens,
            melody_tokens,
            by_bar_melody_tokens,
            by_bar_track_tokens,
        ) = make_data_melodic_fidelity_figaro(filename)

        monophonic_track_tokens_common = dict(
            (k, convert_figaro_to_common(v)) for k, v in monophonic_track_tokens.items()
        )
        melody_tokens_common = convert_musemorphose_to_common(melody_tokens)
        by_bar_melody_tokens_common = [
            convert_musemorphose_to_common(l) for l in by_bar_melody_tokens
        ]
        by_bar_track_tokens_common = [
            dict((k, convert_figaro_to_common(v)) for k, v in d.items())
            for d in by_bar_track_tokens
        ]

        full_seq_fidelity = compute_full_sequence_melodic_fidelity(
            monophonic_track_tokens_common,
            melody_tokens_common,
            by_bar_melody_tokens_common,
            by_bar_track_tokens_common,
        )
        per_beat_fidelity, intersect = compute_by_beat_melodic_fidelity(
            by_bar_melody_tokens_common, by_bar_track_tokens_common
        )

        all_full_seq_fidelity.append(full_seq_fidelity)
        all_per_beat_fidelity.append(per_beat_fidelity)
        all_per_beat_intersect.append(intersect)

    print()
    print(
        "Full sequence fidelity (\N{UPWARDS ARROW}): {:.3f} +/- {:.2f}".format(
            np.mean(all_full_seq_fidelity), np.std(all_full_seq_fidelity)
        )
    )
    print(
        "Per beat fidelity (\N{UPWARDS ARROW}): {:.3f}".format(
            np.mean(all_per_beat_fidelity)
        )
    )
    print(
        "Per beat intersect (\N{UPWARDS ARROW}): {:.3f}".format(
            np.mean(all_per_beat_intersect)
        )
    )
