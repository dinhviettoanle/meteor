import os
from pathlib import Path
import shutil
import numpy as np
from .evaluate_fun import compute_cosine_sim_orig_gen
import argparse

import json

from .evaluate_fun import (
    compute_by_beat_melodic_fidelity,
    compute_full_sequence_melodic_fidelity,
)
from .utils import pickle_load


if __name__ == "__main__":
    from src.utils import pickle_load
    from .attributes import (
        analyze_local_attributes as attributes_analyze_local_attributes,
    )  # type: ignore
    from .attributes import proc_one as attributes_proc_one
    from .custom_data.analyzer import main as analyzer_main
    from .custom_data.corpus2events import main as corpus2events_main
    from .custom_data.events2musemorphose import main as events2musemorphose_main
    from .custom_data.midi2corpus import main as midi2corpus_main


temp_dir = "temp5"


def a_posteriori_analysis(midi_file, folder_analysis="temp_analysis", temporary=True):

    if temporary:
        if os.path.exists(folder_analysis):
            shutil.rmtree(folder_analysis)
        os.makedirs(folder_analysis)
        os.makedirs(os.path.join(folder_analysis, "midi_raw"), exist_ok=True)
        shutil.copy2(midi_file, os.path.join(folder_analysis, "midi_raw"))

    os.makedirs(os.path.join(folder_analysis, "midi_analyzed"))
    os.makedirs(os.path.join(folder_analysis, "corpus"))
    os.makedirs(os.path.join(folder_analysis, "events"))
    os.makedirs(os.path.join(folder_analysis, "remi"))

    analyzer_main(
        path_indir=os.path.join(folder_analysis, "midi_raw"),
        path_outdir=os.path.join(folder_analysis, "midi_analyzed"),
        VERBOSE=False,
    )

    midi2corpus_main(
        path_indir=os.path.join(folder_analysis, "midi_analyzed"),
        path_outdir=os.path.join(folder_analysis, "corpus"),
        multi_track=True,
        multi_track_mapping="mmt",
        VERBOSE=False,
    )

    corpus2events_main(
        path_indir=os.path.join(folder_analysis, "corpus"),
        path_outdir=os.path.join(folder_analysis, "events"),
        path_maskdir=os.path.join(folder_analysis, "melody"),
        multi_track=True,
        pitch_encoding="absolute",
        header_chords=True,
        header_repeatability=True,
        header_pitch_range=True,
        melodic_tokens="track",
        melodic_tokens_position="before",
        order_first="beat",
        VERBOSE=False,
    )

    events2musemorphose_main(
        events_path=os.path.join(folder_analysis, "events"),
        musemorphose_path=os.path.join(folder_analysis, "remi"),
        VERBOSE=False,
    )

    computed_attributes = attributes_proc_one(
        "1.pkl", data_dir=os.path.join(folder_analysis, "remi")
    )

    local_attributes = attributes_analyze_local_attributes(
        "1.pkl", data_dir=os.path.join(folder_analysis, "events")
    )

    #   if temporary:
    #     shutil.rmtree(folder_analysis)

    return {
        "rfreq_cls": computed_attributes["rfreq_cls"],
        "polyph_cls": computed_attributes["polyph_cls"],
        "harmdiv_cls": computed_attributes["harmdiv_cls"],
        "rhythmdiv_cls": computed_attributes["rhythmdiv_cls"],
        "local_attributes": local_attributes,
    }


def analyze_from_midi(midi_path):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    os.makedirs(os.path.join(temp_dir, "midi_raw"))
    shutil.copy2(midi_path, os.path.join(temp_dir, "midi_raw"))

    a_posteriori_analysis(None, folder_analysis=temp_dir, temporary=False)

    return temp_dir


def analyze_generation(arrangement_file):
    analyze_from_midi(arrangement_file)

    tokens = pickle_load(os.path.join(temp_dir, "events/1.pkl"))
    tokens_str = [f"{e['name']}_{e['value']}" for e in tokens]

    with open(arrangement_file.replace(".mid", ".txt"), "w") as f:
        for line in tokens_str:
            f.write(f"{line}\n")

    shutil.rmtree(temp_dir)


def compute_cosine_arrangement_orig_accomontage(eval_files, skip_nomel=False):
    all_sim = []
    orig_filename = list(Path(eval_files).glob("**/*_orig.txt"))[0]
    for gen_filename in list(
        Path(eval_files).glob("**/accomontage_arrangement_band-*.txt")
    ):
        if "_nomel" in str(gen_filename) and skip_nomel:
            continue
        sim = compute_cosine_sim_orig_gen(orig_filename, gen_filename)
        all_sim.append(sim)
    if len(all_sim) > 0:
        return np.nanmean(all_sim)
    else:
        return np.nan


def make_all_cosine_sim_accomontage(eval_files, analyze=False, skip_nomel=False):
    all_sim = []
    all_stuff = list(Path(eval_files).glob("*"))
    for root_piece in sorted(all_stuff):
        if analyze:
            print(root_piece)
            for arrangement_file in root_piece.glob(
                "accomontage_arrangement_band-*.mid"
            ):
                # if Path(str(arrangement_file).replace('.mid', '.txt')).is_file(): continue
                analyze_generation(str(arrangement_file))
        cosine_sim = compute_cosine_arrangement_orig_accomontage(
            root_piece, skip_nomel=skip_nomel
        )
        if analyze:
            print(cosine_sim)
        all_sim.append(cosine_sim)

    print(
        f"W/o error: {np.count_nonzero(~np.isnan(all_sim))} / {len(all_stuff)}",
    )
    return np.nanmean(all_sim), np.nanstd(all_sim)


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


def make_data_melodic_fidelity_accomontage(filename, folder):
    with open(f"{filename}.txt") as fp:
        data_tokens = fp.read().splitlines()

    dict_by_track_seq = dict()
    # Get all tracks
    for token in data_tokens:
        if token.startswith("Track"):
            track_value = token.split("_")[-1]
            dict_by_track_seq[track_value] = []

    # Get track-wise token sequence
    current_track = None
    for token in data_tokens:
        if token.startswith("Track"):
            current_track = token.split("_")[-1]
        elif token.startswith("Bar") or token.startswith("Beat"):
            for _, l in dict_by_track_seq.items():
                l.append(token)
        elif token.startswith("Note") and not (
            "Velocity" in token
        ):  # and not('Duration' in token)
            dict_by_track_seq[current_track].append(token)

    # Convert into monophonic tracks (skyline)
    monophonic_track_tokens = dict((trk, []) for trk in dict_by_track_seq.keys())
    for track, track_tokens in dict_by_track_seq.items():
        beat_content = []
        for ii, token in enumerate(track_tokens):
            if token.startswith("Beat") or token.startswith("Bar"):
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

    # MELODY GROUNTRUTH
    # Get the same number of bars
    piece_id = folder.name.split("_")[0].replace("id", "")
    bar_number = int(folder.name.split("_")[1].replace("bar", ""))
    # mid_to_idx = json.load(open("custom_data/files_symphonynet_full.json"))
    # idx = [(k, v) for k, v in mid_to_idx.items() if f"/{piece_id}" in v][0]

    # melody_filename = os.path.join('custom_data', 'symphonynet_full_multitrack_melodyfirst', 'melody', f'{idx[0]}.pkl')
    melody_filename = os.path.join(
        "/mnt/nfs_share_magnet2/ldinhvie/musemorphose_data/symphonynet_full_multitrack_melfull/melody",
        f"{piece_id}.pkl",
    )
    melody_events = pickle_load(melody_filename)["melody_events"]

    chunk_melody_events = []
    by_bar_melody_events = []
    mel_n_bars = -1  # Shift somewhere, didnt' get it...
    for event in melody_events:
        if event["name"] == "Bar":
            mel_n_bars += 1
        if mel_n_bars >= bar_number:
            if event["name"] == "Bar" and len(by_bar_melody_events) > 0:
                chunk_melody_events.append(by_bar_melody_events)
                by_bar_melody_events = []
            elif event["name"] != "Bar":
                by_bar_melody_events.append(event)
        if mel_n_bars - bar_number == n_bars:
            break

    # Make melody tokens dict
    # print([x for x in chunk_melody_events[0] if x['name'] == 'Note_Pitch'])
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
            ]  # and 'Duration' not in x
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


def melodic_similarity_accomontage(path, skip_nomel=True):
    all_full_seq_fidelity = []
    all_per_beat_fidelity = []
    all_per_beat_intersect = []

    path = Path(path)
    for folder in sorted(list(path.glob("*"))):  # id10859_bar34_orig
        for ii, filename in enumerate(
            folder.glob("**/accomontage_arrangement_band-*.mid")
        ):
            if skip_nomel and not ("nomel" in filename.name):
                continue
            if not (skip_nomel) and ("nomel" in filename.name):
                continue
            filename = str(filename).replace(".mid", "")
            # print(filename)
            try:
                (
                    monophonic_track_tokens,
                    melody_tokens,
                    by_bar_melody_tokens,
                    by_bar_track_tokens,
                ) = make_data_melodic_fidelity_accomontage(filename, folder)
                full_seq_fidelity = compute_full_sequence_melodic_fidelity(
                    monophonic_track_tokens,
                    melody_tokens,
                    by_bar_melody_tokens,
                    by_bar_track_tokens,
                    VERBOSE=False,
                )
                per_beat_fidelity, intersect = compute_by_beat_melodic_fidelity(
                    by_bar_melody_tokens, by_bar_track_tokens
                )
                all_full_seq_fidelity.append(full_seq_fidelity)
                all_per_beat_fidelity.append(per_beat_fidelity)
                all_per_beat_intersect.append(intersect)
            except Exception as e:
                print()
                print(">>> Error on {}: {}".format(filename, e))

    print()
    if np.isnan(all_full_seq_fidelity).any():
        print("Warning: nan in fidelities")
    print(
        "Full sequence fidelity (\N{UPWARDS ARROW}): {:.3f} +/- {:.2f}".format(
            np.nanmean(all_full_seq_fidelity), np.nanstd(all_full_seq_fidelity)
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


# def tokenize_accomontage_gen(eval_files):


if __name__ == "__main__":
    # eval_files = '/mnt/nfs_share_magnet2/ldinhvie/accomontage3/leadsheet_skyline_gen_enforced/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    eval_files = args.folder
    make_all_cosine_sim_accomontage(eval_files, analyze=True)
