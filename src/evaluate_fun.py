import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import miditoolkit
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm.auto import tqdm

from .custom_data.corpus2events import make_pitchclass_token

from .remi2midi import pitchclass_octave_to_midi
from .utils import pickle_dump

plt.rcParams["image.cmap"] = "Blues"

# ================================================================================================================
# =========================================== BAR CONTROLLABILITY ================================================
# ================================================================================================================


def plot_confusion(query, analysis, title):
    print(classification_report(query, analysis))
    disp = ConfusionMatrixDisplay(
        confusion_matrix(query, analysis, normalize="true",
                         labels=np.arange(0, 8))
    )
    disp.plot()
    plt.ylabel("Query class")
    plt.xlabel("A posteriori analysis class")
    plt.title(title)
    plt.show()


def make_all_barcontrols(perf_files):
    list_perfo = []

    for perf_file in perf_files:
        with open(perf_file, "r") as f:
            data = json.load(f)
        perf_data = dict()
        for k, v in data["query"].items():
            perf_data[f"query_{k}"] = v
        for k, v in data["analysis"].items():
            perf_data[f"analysis_{k}"] = v[:-1]
        perf_data["entropy"] = data.get("entropy")
        perf_data["inference_time"] = data.get("inference_time")
        list_perfo.append(perf_data)

    df_perfo = pd.DataFrame(list_perfo)
    df_perfo["query_len"] = df_perfo.query_rfreq_cls.apply(len)
    df_perfo["analysis_len"] = df_perfo.analysis_rfreq_cls.apply(len)

    for column_name in df_perfo.columns:
        if "cls" in column_name:
            df_perfo[column_name] = df_perfo.apply(
                lambda row: row[column_name][: min(
                    row.analysis_len, row.query_len)],
                axis=1,
            )

    try:
        mean_entropy = df_perfo.entropy.mean()
    except TypeError:
        mean_entropy = df_perfo.entropy.apply(lambda x: np.mean(x)).mean()

    print("Entropy:", mean_entropy)
    print("Inference time:", df_perfo.inference_time.mean())

    query = np.concatenate(df_perfo.query_rfreq_cls)
    analysis = np.concatenate(df_perfo.analysis_rfreq_cls)
    print("Spearman:", spearmanr(query, analysis))
    query_rhyhtm, analysis_rhythm = query, analysis

    plot_confusion(query, analysis, "Rhythmicity")

    query = np.concatenate(df_perfo.query_polyph_cls)
    analysis = np.concatenate(df_perfo.analysis_polyph_cls)
    print("Spearman:", spearmanr(query, analysis))
    query_polyph, analysis_polyph = query, analysis

    plot_confusion(query, analysis, "Polyphonicity")

    print("Spearman poly|rhythm", spearmanr(query_rhyhtm, analysis_polyph))
    print("Spearman rhythm|poly", spearmanr(query_polyph, analysis_rhythm))


# ================================================================================================================
# =========================================== CHROMA SIMILARITY ==================================================
# ================================================================================================================


def cosine_sim(p_true, p_pred):
    return np.sum(p_true * p_pred)


def chroma(events):
    pitch_classes = [item % 12 for item in events]
    if len(pitch_classes):
        count = np.bincount(pitch_classes, minlength=12)
        count = count / np.sqrt(np.sum(count**2))
    else:
        count = np.array([1 / 12] * 12)
    return count


def make_chroma_vectors(filename):
    with open(f"{filename}") as fp:
        data_tokens = fp.read().splitlines()

    all_pitches = []
    pitches_in_bar = []
    for token in data_tokens:
        if "Bar" in token and len(pitches_in_bar) > 0:
            all_pitches.append(pitches_in_bar)
            pitches_in_bar = []
        elif "Note_Pitch" in token:
            pitches_in_bar.append(int(token.split("_")[-1]))

    return all_pitches


def compute_cosine_sim_orig_gen(orig_filename, gen_filename):
    piece_orig_chroma = make_chroma_vectors(orig_filename)
    piece_gen_chroma = make_chroma_vectors(gen_filename)
    all_cosine_dist = []
    for bar_orig_chroma, bar_gen_chroma in zip(piece_orig_chroma, piece_gen_chroma):
        vect_orig = chroma(bar_orig_chroma)
        vect_gen = chroma(bar_gen_chroma)
        cosine = cosine_sim(vect_orig, vect_gen)
        all_cosine_dist.append(cosine)
    if len(all_cosine_dist) > 0:
        return np.mean(all_cosine_dist)
    else:
        return 0


def make_all_cosine_sim(eval_files):
    all_sim = []
    for orig_filename in list(Path(eval_files).glob("**/*_orig.txt")):
        orig_prefix = orig_filename.stem.replace("_orig", "")
        for gen_filename in list(Path(eval_files).glob(f"**/{orig_prefix}*.txt")):
            sim = compute_cosine_sim_orig_gen(orig_filename, gen_filename)
            all_sim.append(sim)
    return np.mean(all_sim), np.std(all_sim)


# ================================================================================================================
# =========================================== TRACK CONTROLLABILITY - PitchRange =================================
# ================================================================================================================


def make_all_pitchrange(perf_files):
    corpus_query = []
    corpus_analysis = []
    corpus_spearman = []
    corpus_accuracy = []

    for perf_file in perf_files:
        with open(perf_file, "r") as f:
            data = json.load(f)

        list_query = []
        for bar_data in data["query"]["local_attributes"]["pitchrange"]:
            bar_query = {}
            for token in bar_data:
                data_instr_value = token.replace("Description_PitchRange_", "").split(
                    "-"
                )
                instr = "-".join(data_instr_value[:-1])
                value = data_instr_value[-1]
                bar_query[instr] = value
            list_query.append(bar_query)

        list_analysis = []
        for bar_data in data["analysis"]["local_attributes"]:
            bar_analysis = {}
            for instr, value in bar_data["pitchrange"]:
                bar_analysis[instr] = value
            list_analysis.append(bar_analysis)

        list_accuracy = []
        list_spearman = []
        piece_query = []
        piece_analysis = []

        for ii_bar, (all_query, all_analysis) in enumerate(
            zip(list_query, list_analysis)
        ):
            query = []
            analysis = []
            for instr in all_query.keys():
                if instr not in all_analysis:
                    continue
                query.append(int(all_query[instr]))
                analysis.append(int(all_analysis[instr]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                list_accuracy.append(accuracy_score(query, analysis))
            # list_spearman.append(spearmanr(query, analysis))
            piece_query += query
            piece_analysis += analysis

        corpus_query += piece_query
        corpus_analysis += piece_analysis
        corpus_accuracy += list_accuracy
        corpus_spearman += list_spearman

    print("Spearman:", spearmanr(corpus_query, corpus_analysis))
    print(classification_report(corpus_query, corpus_analysis, zero_division=0))
    plt.rcParams.update({"font.size": 8})
    disp = ConfusionMatrixDisplay(
        confusion_matrix(
            corpus_query,
            corpus_analysis,
            normalize="true",
            labels=np.arange(0, 130, 10),
        )
    )
    disp.plot()
    plt.ylabel("Query class")
    plt.xlabel("A posteriori analysis class")
    plt.show()


# ================================================================================================================
# =========================================== TRACK CONTROLLABILITY - Repeatability ==============================
# ================================================================================================================


def make_all_repeatability(perf_files):
    corpus_query = []
    corpus_analysis = []
    corpus_spearman = []
    corpus_accuracy = []

    for perf_file in perf_files:
        with open(perf_file, "r") as f:
            data = json.load(f)

        list_query = []
        for bar_data in data["query"]["local_attributes"]["repeatability"]:
            bar_query = {}
            for token in bar_data:
                data_instr_value = token.replace(
                    "Description_Repeatability_", ""
                ).split("-")
                instr = "-".join(data_instr_value[:-1])
                value = data_instr_value[-1]
                bar_query[instr] = value
            list_query.append(bar_query)

        list_analysis = []
        for bar_data in data["analysis"]["local_attributes"]:
            bar_analysis = {}
            for instr, value in bar_data["repeatability"]:
                bar_analysis[instr] = value
            list_analysis.append(bar_analysis)

        list_accuracy = []
        list_spearman = []
        piece_query = []
        piece_analysis = []

        for ii_bar, (all_query, all_analysis) in enumerate(
            zip(list_query, list_analysis)
        ):
            query = []
            analysis = []
            for instr in all_query.keys():
                if instr not in all_analysis:
                    continue
                query.append(int(all_query[instr]))
                analysis.append(int(all_analysis[instr]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                list_accuracy.append(accuracy_score(query, analysis))
            # list_spearman.append(spearmanr(query, analysis))
            piece_query += query
            piece_analysis += analysis

        corpus_query += piece_query
        corpus_analysis += piece_analysis
        corpus_accuracy += list_accuracy
        corpus_spearman += list_spearman

    print("Spearman:", spearmanr(corpus_query, corpus_analysis))
    print(classification_report(corpus_query, corpus_analysis, zero_division=0))
    plt.rcParams.update({"font.size": 8})
    disp = ConfusionMatrixDisplay(
        confusion_matrix(
            corpus_query, corpus_analysis, normalize="true", labels=np.arange(0, 13)
        )
    )
    disp.plot()
    plt.ylabel("Query class")
    plt.xlabel("A posteriori analysis class")
    plt.show()


# ================================================================================================================
# =========================================== MELODIC FIDELITY ===================================================
# ================================================================================================================


def token_edit_levenstein_similarity_normalized(text1: str, text2: str) -> float:
    """
    Compute the normalized levenstein distance between two texts.
    """
    return nltk.edit_distance(text1, text2) / max(len(text1), len(text2))


def make_data_melodic_fidelity(filename):
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
        # and not('Duration' in token)
        elif token.startswith("Note") and not ("Velocity" in token):
            dict_by_track_seq[current_track].append(token)

    # Convert into monophonic tracks (skyline)
    monophonic_track_tokens = dict((trk, [])
                                   for trk in dict_by_track_seq.keys())
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
        single_bar_by_tracks = dict((trk, [])
                                    for trk in dict_by_track_seq.keys())
        for trk in single_bar_by_tracks.keys():
            single_bar_by_tracks[trk] = track_tokens_by_bar[trk][ii]
        by_bar_track_tokens.append(single_bar_by_tracks)

    # GROUNDTRUTH MELODY
    with open(f"{filename}_melody_tokens.json") as fp:
        melody_tokens_dict = json.load(fp)

    # Full sequence
    melody_tokens = []
    by_bar_melody_tokens = []
    for bar, bar_content in melody_tokens_dict.items():
        melody_tokens.append("Bar_None")
        bar_tokens = []
        for beat, beat_content in bar_content.items():
            # and 'Duration' not in x
            beat_content_no_vel = [
                x for x in beat_content if "Velocity" not in x]
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


def compute_full_sequence_melodic_fidelity(
    monophonic_track_tokens,
    melody_tokens,
    by_bar_melody_tokens,
    by_bar_track_tokens,
    VERBOSE=False,
):
    if VERBOSE:
        print("FULL SEQUENCE")
    for track_name, track_tokens in monophonic_track_tokens.items():
        if VERBOSE:
            print(
                "{:<20}{:.5f} {}".format(
                    track_name,
                    token_edit_levenstein_similarity_normalized(
                        melody_tokens, track_tokens
                    ),
                    nltk.edit_distance(melody_tokens, track_tokens),
                )
            )

    if VERBOSE:
        print()

    all_minimums_edit_distance = []
    for ii_bar, (mel_tokens, trk_tokens) in enumerate(
        zip(by_bar_melody_tokens, by_bar_track_tokens)
    ):
        if VERBOSE:
            print(f"Bar {ii_bar}")
        all_trk_edit_distance = []
        for track_name, track_tokens in trk_tokens.items():
            if len(track_tokens) == 0 or len(mel_tokens) == 0:
                continue
            norm_edit_distance = token_edit_levenstein_similarity_normalized(
                mel_tokens, track_tokens
            )
            if VERBOSE:
                print(
                    "{:<20}{:.5f} {}".format(
                        track_name,
                        norm_edit_distance,
                        nltk.edit_distance(mel_tokens, track_tokens),
                    )
                )
            all_trk_edit_distance.append((norm_edit_distance, track_name))
        if len(all_trk_edit_distance) > 0:
            if VERBOSE:
                print("Minimum edit distance = {}".format(
                    min(all_trk_edit_distance)))
            all_minimums_edit_distance.append(min(all_trk_edit_distance))
        if VERBOSE:
            print()

    average_fidelity = np.mean([1 - x[0] for x in all_minimums_edit_distance])
    if VERBOSE:
        print("Average minimal edit distance")
        print(average_fidelity)
        print()

    return average_fidelity


def compute_by_beat_melodic_fidelity(
    by_bar_melody_tokens, by_bar_track_tokens, VERBOSE=False
):
    # Melody
    melody_beat_tokens = [[] for _ in range(len(by_bar_melody_tokens))]
    for ii_bar, bar_tokens in enumerate(by_bar_melody_tokens):
        beat_tokens = []
        for token in bar_tokens:
            if token.startswith("Beat"):
                if len(beat_tokens) > 1:
                    melody_beat_tokens[ii_bar].append("-".join(beat_tokens))
                beat_tokens = []
            beat_tokens.append(token)
        melody_beat_tokens[ii_bar].append("-".join(beat_tokens))

    # Tracks
    track_beat_tokens = [dict() for _ in range(len(by_bar_track_tokens))]
    for ii_bar, dict_tracks in enumerate(by_bar_track_tokens):
        for track, bar_tokens in dict_tracks.items():
            beat_tokens = []
            for token in bar_tokens:
                if token.startswith("Beat"):
                    if len(beat_tokens) > 1:
                        if track not in track_beat_tokens[ii_bar]:
                            track_beat_tokens[ii_bar][track] = []
                        track_beat_tokens[ii_bar][track].append(
                            "-".join(beat_tokens))
                    beat_tokens = []
                beat_tokens.append(token)

            if len(beat_tokens) > 1:
                if track not in track_beat_tokens[ii_bar]:
                    track_beat_tokens[ii_bar][track] = []
                track_beat_tokens[ii_bar][track].append("-".join(beat_tokens))

    if VERBOSE:
        print("BY BEAT")
    all_minimums_edit_distance = []
    all_maximums_intersect = []
    for ii_bar, (mel_tokens, trk_tokens) in enumerate(
        zip(melody_beat_tokens, track_beat_tokens)
    ):
        if VERBOSE:
            print(f"Bar {ii_bar}")
        all_trk_edit_distance = []
        all_trk_intersect = []
        for track_name, track_tokens in trk_tokens.items():
            norm_edit_distance = token_edit_levenstein_similarity_normalized(
                mel_tokens, track_tokens # type: ignore
            )
            intersect = len(list(set(mel_tokens) & set(
                track_tokens))) / len(mel_tokens)
            if VERBOSE:
                print(
                    "{:<20}{:.5f} {:<3} {}".format(
                        track_name,
                        norm_edit_distance,
                        nltk.edit_distance(mel_tokens, track_tokens),
                        intersect,
                    )
                )
            all_trk_edit_distance.append((norm_edit_distance, track_name))
            all_trk_intersect.append((intersect, track_name))

        min_edit_distance = (
            min(all_trk_edit_distance) if len(
                all_trk_edit_distance) > 0 else None
        )
        max_trk_intersect = (
            max(all_trk_intersect) if len(all_trk_intersect) > 0 else None
        )
        if VERBOSE:
            print("Minimum edit distance = {}".format(min_edit_distance))
        if VERBOSE:
            print("Maximum intersect = {}".format(max_trk_intersect))
        if len(all_trk_edit_distance) > 0:
            all_minimums_edit_distance.append(min(all_trk_edit_distance))
            all_maximums_intersect.append(max(all_trk_intersect))
        if VERBOSE:
            print()

    average_fidelity = np.mean([1 - x[0] for x in all_minimums_edit_distance])
    if VERBOSE:
        print("Average fidelity")
        print(average_fidelity)
        print("Average maximal intersect")
        print(np.mean([x[0] for x in all_maximums_intersect]))

    return average_fidelity, np.mean([x[0] for x in all_maximums_intersect])


def make_all_melodic_fidelity(path):
    all_full_seq_fidelity = []
    all_per_beat_fidelity = []
    all_per_beat_intersect = []

    for ii, filename in enumerate(Path(path).glob("**/*.mid")):
        if "orig" in filename.name:
            continue
        filename = str(filename).replace(".mid", "")
        # print(filename)
        print(ii, end=" ")
        try:
            (
                monophonic_track_tokens,
                melody_tokens,
                by_bar_melody_tokens,
                by_bar_track_tokens,
            ) = make_data_melodic_fidelity(filename)
            full_seq_fidelity = compute_full_sequence_melodic_fidelity(
                monophonic_track_tokens,
                melody_tokens,
                by_bar_melody_tokens,
                by_bar_track_tokens,
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


# ================================================================================================================
# =========================================== MELODIC PITCH RANGE PER INSTRUMENT =================================
# ================================================================================================================


def make_avg_pitch(filename, mel_instrument, VERBOSE=False):
    with open(f"{filename}") as fp:
        data_tokens = fp.read().splitlines()

    for token in data_tokens:
        dict_by_track_seq = dict()
        # Get all tracks
        for token in data_tokens:
            if token.startswith("Track"):
                track_value = token.split("_")[-1]
                dict_by_track_seq[track_value] = []

    current_track = None
    for token in data_tokens:
        if token.startswith("Track"):
            current_track = token.split("_")[-1]
        # elif token.startswith('Bar') or token.startswith('Beat'):
        # for _, l in dict_by_track_seq.items():
        #     l.append(token)
        # and not('Duration' in token)
        elif token.startswith("Note_PitchClass") or token.startswith("Note_Octave"):
            dict_by_track_seq[current_track].append(token)

    mel_notes = dict_by_track_seq[mel_instrument]
    mel_midis = []

    for pc_token, octave_token in zip(mel_notes[::2], mel_notes[1::2]):
        pc_value = pc_token.split("_")[-1]
        octave_value = int(octave_token.split("_")[-1])
        mel_midis.append(pitchclass_octave_to_midi(pc_value, octave_value))

    mean_midi = np.mean(mel_midis)
    std_midi = np.std(mel_midis)
    if VERBOSE:
        print(
            "Avg note: ({}) {}".format(
                int(mean_midi), make_pitchclass_token(int(mean_midi))
            )
        )
        # print("Note - std:", make_pitchclass_token(int(mean_midi - std_midi)))
        # print("Note + std:", make_pitchclass_token(int(mean_midi + std_midi)))
    return mel_midis, mean_midi, std_midi


def make_all_avg_pitch(token_files, mel_instrument, limits):
    all_mel_midis = []
    for ii, filename in enumerate(token_files):
        if "orig" in filename.name:
            continue
        # print(filename, end=' ')
        print(ii, end=" ")
        mel_midis, mean_midi, std_midi = make_avg_pitch(
            filename, mel_instrument)
        all_mel_midis += mel_midis

    all_mel_midis = np.array(all_mel_midis)
    below_range = len(np.where(all_mel_midis < limits[0])[0])
    above_range = len(np.where(all_mel_midis > limits[1])[0])
    mean_midi = np.mean(all_mel_midis)
    std_midi = np.std(all_mel_midis)
    print("===================================")
    print(
        "Avg note: ({}) {}".format(
            int(mean_midi), make_pitchclass_token(int(mean_midi))
        )
    )
    print("Note - std:", make_pitchclass_token(int(mean_midi - std_midi)))
    print("Note + std:", make_pitchclass_token(int(mean_midi + std_midi)))
    print(
        "Outside range: {:.1%}".format(
            (below_range + above_range) / len(all_mel_midis))
    )


# ============================================ INSTRUMENTATION REALISTICNESS =========================================


def make_pitch_distribution_from_midi(midi_file):
    count_notes = dict(
        [(track, dict([(pitch, 0) for pitch in range(128)]))
         for track in range(128)]
    )
    try:
        midi_obj = miditoolkit.MidiFile(midi_file)
        for track in midi_obj.instruments:
            for note in track.notes:
                count_notes[track.program][note.pitch] += 1 # type: ignore
    except Exception:
        print("Error with", midi_file)
    return count_notes


def make_pitch_distribution_from_folder(midi_files):
    df_counts = pd.DataFrame({"midi_file": midi_files})

    df_counts["pitch_distrib"] = df_counts.midi_file.parallel_apply(
        make_pitch_distribution_from_midi
    )

    # Initialize a merged dictionary
    merged_counts = dict(
        [(track, dict([(pitch, 0) for pitch in range(128)]))
         for track in range(128)]
    )

    for ii, row in tqdm(df_counts.iterrows(), total=len(df_counts)):
        pitch_distrib = row.pitch_distrib
        for track in range(128):
            for pitch in range(128):
                merged_counts[track][pitch] += pitch_distrib[track][pitch]

    return merged_counts


if __name__ == "__main__":
    dataset_distrib = make_pitch_distribution_from_folder(
        list(Path("/mnt/nfs_share_magnet2/ldinhvie/lmd_full").glob("**/*.mid"))
    )
    pickle_dump(dataset_distrib, "lmd_distrib.pickle")
