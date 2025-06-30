"""
intsr_gird = {
    track_1: {
        time0: [events]
        time1: [events]
    }
    track_2: {
        time0: [events]
        time1: [events]
    }
}
"""

import collections
import json
import os
import pickle
from pprint import pprint

import miditoolkit
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from .constants import GEMINI_INSTRUMENT_SET, GEMINI_PREDICT, MMT_MAPPING_PATH
from .utils import traverse_dir

pandarallel.initialize(progress_bar=True)

# ================================================== #
#  Configuration                                     #
# ================================================== #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

# INSTR_NAME_MAP = {"piano": 0}
# Gemini mapping for SOD
GEMINI_PREDICT = dict(
    (k.encode("ascii", "ignore"), v) for k, v in GEMINI_PREDICT.items()
)
INSTR_NAME_MAP = dict((instr, instr) for ii, instr in enumerate(GEMINI_INSTRUMENT_SET))
# MMT mappign (general)
with open(MMT_MAPPING_PATH) as f:
    mmt_mapping = json.load(f)

MMT_PROGRAM_TO_INSTRUMENT = dict(
    (int(c), instr) for c, instr in mmt_mapping["program_instrument_map"].items()
)

MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1  #  0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 64 + 1, dtype=np.int32)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=np.int32)
DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=np.int32)
DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 8, BEAT_RESOL * 8 + 1, BEAT_RESOL / 8)

# path_indir_default = "./midi_analyzed"
path_indir_default = "./symphonynet_classical_analyzed"
# path_indir_default = "./symphonynet_classical_midi"

# ================================================== #


def get_instr_name(track_name, track_program, multi_track_mapping):
    if multi_track_mapping is None:
        raise AttributeError("You must provide a multi_track_mapping")

    elif multi_track_mapping == "sod_gemini":
        prediction = GEMINI_PREDICT[track_name.encode("ascii", "ignore")]
        if prediction == "Unknown":
            return None
        return INSTR_NAME_MAP[prediction]

    elif multi_track_mapping == "mmt":
        return MMT_PROGRAM_TO_INSTRUMENT[track_program]



def proc_one(
    path_midi, path_outfile, verbose=False, multi_track=True, multi_track_mapping=None
):
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi) # type: ignore
    # print(midi_obj.instruments)
    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # print(instr)
        if instr.is_drum:
            continue  # skip drum tracks
        if multi_track:
            instr_idx = get_instr_name(instr.name, instr.program, multi_track_mapping)
            if not (instr_idx):
                continue
        else:  # monotrack
            instr_idx = "piano"  # INSTR_NAME_MAP[instr.name]

        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(" [x] Unknown type of sorting.")

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if (
            marker.text.split("_")[0] != "global"
            and "Boundary" not in marker.text.split("_")[0]
        ):
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if "Boundary" in marker.text.split("_")[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split("_")[0] == "global" and marker.text.split("_")[1] == "bpm":
            gobal_bpm = int(marker.text.split("_")[2])

    # --- process items to grid --- #
    # compute empty bar offset at head
    ## VT: Fix if MIDI not produced with 480 BEAT_RESOL
    first_note_time = min(
        [
            instr_notes[k][0].start * BEAT_RESOL / midi_obj.ticks_per_beat
            for k in instr_notes.keys()
        ]
    )
    last_note_time = max(
        [
            instr_notes[k][-1].start * BEAT_RESOL / midi_obj.ticks_per_beat
            for k in instr_notes.keys()
        ]
    )

    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL  # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    if verbose:
        print(" > offset:", offset)
        print(" > last_bar:", last_bar)

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL
            ## VT: Fix if MIDI not produced with 480 BEAT_RESOL
            note.start = int(note.start * BEAT_RESOL / midi_obj.ticks_per_beat)
            note.end = int(note.end * BEAT_RESOL / midi_obj.ticks_per_beat)

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))
            ]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time
            note.shift = DEFAULT_SHIFT_BINS[
                np.argmin(abs(DEFAULT_SHIFT_BINS - note.shift))
            ]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)

        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time = 0 if chord.time < 0 else chord.time
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]

    # process global bpm
    gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - gobal_bpm))]

    # collect
    if verbose:
        print("> instr:", intsr_gird.keys())
    song_data = {
        "notes": intsr_gird,
        "chords": chord_grid,
        "tempos": tempo_grid,
        "labels": label_grid,
        "metadata": {
            "global_bpm": gobal_bpm,
            "last_bar": last_bar,
        },
    }

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[: -len(fn)], exist_ok=True)
    pickle.dump(song_data, open(path_outfile, "wb"))


def main(
    path_indir=path_indir_default,
    path_outdir="./corpus",
    multi_track=True,
    multi_track_mapping=None,
    VERBOSE=True,
):
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(path_indir, is_pure=True, is_sort=True)
    n_files = len(midifiles)
    if VERBOSE:
        print("num files:", n_files)

    # run all
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        if VERBOSE:
            print("{}/{}: {}".format(fidx, n_files, path_midi))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, str(fidx + 1) + ".pkl")

        # proc
        proc_one(
            path_infile,
            path_outfile,
            multi_track=multi_track,
            multi_track_mapping=multi_track_mapping,
        )


def main_parallel(
    path_indir=path_indir_default,
    path_outdir="./corpus",
    multi_track=True,
    multi_track_mapping=None,
    files_limit=None,
):
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(path_indir, is_pure=True, is_sort=True)
    n_files = len(midifiles) if files_limit is None else files_limit
    print("num files:", n_files)

    list_data = []
    # run all
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print("{}/{}: {}".format(fidx + 1, n_files, path_midi))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, str(fidx + 1) + ".pkl")
        list_data.append(
            {
                "path_infile": path_infile,
                "path_outfile": path_outfile,
                "multi_track": multi_track,
                "multi_track_mapping": multi_track_mapping,
            }
        )

    df_proc = pd.DataFrame(list_data)
    parallel_proc_one = lambda row: proc_one(
        row.path_infile,
        row.path_outfile,
        multi_track=row.multi_track,
        multi_track_mapping=row.multi_track_mapping,
    )
    df_proc.parallel_apply(parallel_proc_one, axis=1) # type: ignore


if __name__ == "__main__":
    main()
