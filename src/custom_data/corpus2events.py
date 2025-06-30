import itertools
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

from .melody_extraction import (
    extract_melody_events,
    extract_melody_track,
    make_melodic_sequence,
)
from .utils import traverse_dir

pandarallel.initialize(progress_bar=True)
# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
MAX_INSTRUMENTS = 16
NOTES_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# path_root_default = "./ailab17k_from-scratch_remi"
path_root_default = "./symphonynet_classical_remi"
# path_root_default = "./test"

INCLUDE_CHORDS_IN_MELODY = False
if INCLUDE_CHORDS_IN_MELODY:
    print("\033[93m/!\\ MELODY TOKENS WILL INCLUDE CHORDS /!\\\033[0m")
else:
    print("\033[93m/!\\ MELODY TOKENS WILL *NOT* INCLUDE CHORDS /!\\\033[0m")


# utilities
def plot_hist(data, path_outfile):
    print(data)
    print("[Fig] >> {}".format(path_outfile))
    data_mean = np.mean(data)
    data_std = np.std(data)

    print("mean:", data_mean)
    print(" std:", data_std)

    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title("mean: {:.3f}_std: {:.3f}".format(data_mean, data_std))
    plt.savefig(path_outfile)
    plt.close()



# define event
def create_event(name, value):
    event = dict()
    event["name"] = name
    event["value"] = value
    return event


def make_pitchclass_token(midi_value):
    pitch_class = NOTES_NAMES[midi_value % 12]
    octave = max(0, midi_value // 12 - 1)
    return create_event("Note_PitchClass", pitch_class), create_event(
        "Note_Octave", octave
    )


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


# ==========================================================================================================


def make_sequence_one_bar_beat_first(
    data: Dict, bar_step: int, instr_idxs: List, multi_track: bool, pitch_encoding: str
) -> Tuple[List, Dict, List]:
    """Make one bar music data, using <Beat> token first, and then, multiple <Track>

    Parameters
    ----------
    data : Dict
        Corpus data
    bar_step : int
        nth bar
    instr_idxs : List
        Instruments included in the data of the bar
    multi_track : bool
        Include track tokens
    pitch_encoding : str
        Pitch encoding (absolute or pitchclass)

    Returns
    -------
    Bar events, pitch ranges per instruments, chords included in the bar
    """
    bar_events = []
    instruments_in_bar = []
    chords_in_bar = []
    instruments_pitch_range = dict()

    for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
        pos_events = []

        # unpack
        t_chords = data["chords"][timing]
        t_tempos = data["tempos"][timing]

        # chord
        if len(t_chords):
            root, quality, bass = t_chords[0].text.split("_")
            chord_value = root + "_" + quality
            pos_events.append(create_event("Chord", chord_value))
            if chord_value not in chords_in_bar:
                chords_in_bar.append(chord_value)

        # tempo
        if len(t_tempos):
            pos_events.append(create_event("Tempo", t_tempos[0].tempo))

        # note
        for instr_idx in instr_idxs:
            t_notes = data["notes"][instr_idx][timing]  # piano track
            if len(t_notes) and sum(note.duration for note in t_notes) > 0:
                if multi_track:
                    pos_events.append(
                        create_event("Track", instr_idx)
                    )  # here or for each note event?

                if instr_idx not in instruments_in_bar:
                    instruments_in_bar.append(instr_idx)
                    instruments_pitch_range[instr_idx] = []

                for note in t_notes:
                    if note.duration == 0:
                        continue
                    if pitch_encoding == "absolute":
                        note_event = (create_event("Note_Pitch", note.pitch),)
                    elif pitch_encoding == "pitchclass":
                        note_event = make_pitchclass_token(note.pitch)
                    else:
                        raise AttributeError(
                            "Wrong pitch encoding provided: {}".format(pitch_encoding)
                        )

                    pos_events.extend(
                        [
                            *note_event,
                            create_event("Note_Velocity", note.velocity),
                            create_event("Note_Duration", note.duration),
                        ]
                    )

                    instruments_pitch_range[instr_idx].append(note.pitch)

        # collect & beat
        if len(pos_events):
            bar_events.append(create_event("Beat", (timing - bar_step) // TICK_RESOL))
            # TODO: Add current chord?
            bar_events.extend(pos_events)

    return bar_events, instruments_pitch_range, chords_in_bar


# ==========================================================================================================


def make_sequence_one_bar_track_first(
    data: Dict, bar_step: int, instr_idxs: List, multi_track: bool, pitch_encoding: str
) -> Tuple[List, Dict, List]:
    """Make one bar music data, using <Track> token first, and then, multiple <Beat>

    Parameters
    ----------
    data : Dict
        Corpus data
    bar_step : int
        nth bar
    instr_idxs : List
        Instruments included in the data of the bar
    multi_track : bool
        Include track tokens (yes...)
    pitch_encoding : str
        Pitch encoding (absolute or pitchclass)

    Returns
    -------
    Bar events, pitch ranges per instruments, chords included in the bar
    """
    bar_events = []
    instruments_in_bar = []
    chords_in_bar = []
    instruments_pitch_range = dict()

    for instr_idx in instr_idxs:
        track_events = []
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):

            # unpack
            t_chords = data["chords"][timing]
            t_tempos = data["tempos"][timing]
            t_notes = data["notes"][instr_idx][timing]  # piano track

            if len(t_notes):
                track_events.append(
                    create_event("Beat", (timing - bar_step) // TICK_RESOL)
                )
            else:
                continue

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split("_")
                chord_value = root + "_" + quality
                track_events.append(create_event("Chord", chord_value))
                if chord_value not in chords_in_bar:
                    chords_in_bar.append(chord_value)

            # tempo
            if len(t_tempos):
                track_events.append(create_event("Tempo", t_tempos[0].tempo))

            if len(t_notes):
                for note in t_notes:
                    if note.duration == 0:
                        continue
                    if pitch_encoding == "absolute":
                        note_event = (create_event("Note_Pitch", note.pitch),)
                    elif pitch_encoding == "pitchclass":
                        note_event = make_pitchclass_token(note.pitch)
                    else:
                        raise AttributeError(
                            "Wrong pitch encoding provided: {}".format(pitch_encoding)
                        )

                    track_events.extend(
                        [
                            *note_event,
                            create_event("Note_Velocity", note.velocity),
                            create_event("Note_Duration", note.duration),
                        ]
                    )

                    if instr_idx not in instruments_in_bar:
                        instruments_in_bar.append(instr_idx)
                        instruments_pitch_range[instr_idx] = []

                    instruments_pitch_range[instr_idx].append(note.pitch)

        # collect & beat
        if len(track_events):
            bar_events.append(create_event("Track", instr_idx))
            bar_events.extend(track_events)

    return bar_events, instruments_pitch_range, chords_in_bar


# ==========================================================================================================
# ==========================================================================================================


# core functions
def corpus2event_remi_v2(
    path_infile: str,
    path_outfile: str,
    path_maskfile: Optional[str] = None,
    multi_track: bool = True,
    header_pitch_range: bool = True,
    header_repeatability: bool = True,
    header_melodic_instr: bool = False,
    header_chords: bool = True,
    pitch_encoding: str = "absolute",
    melodic_tokens: Optional[str] = None,
    melodic_tokens_position: Optional[str] = None,
    order_first: str = "beat",
):
    """Converts a single corpus file to event file

    Parameters
    ----------
    path_infile : str
        Input file
    path_outfile : str
        Output file
    path_maskdir : Optional[str], optional
        Output directory for melodic mask, by default None
    multi_track : bool, optional
        Include track tokens, by default True
    header_pitch_range : bool, optional
        Add pitch_range tokens in the header, by default True
    header_repeatability : bool, optional
        Add reapeatability tokens in the header, by default True
    header_chords : bool, optional
        Add chords tokens in the header, by default True
    header_melodic_instr : bool, optional
        Add melodic_instrument tokens in the header, by default False
    pitch_encoding : str, optional
        Pitch encoding (absolute or pitchclass), by default "absolute"
    melodic_tokens : Optional[str], optional
        Skyline algorithm: (note-wise (ie beat-wise) or track-wise), by default None
    melodic_tokens_position : Optional[str], optional
        Where is the information of melodic events situated (before (ie. in the header) or note), by default None
    order_first : str, optional
        Beat-first or track-first parsing, by default "beat"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    AttributeError
        _description_
    """

    data = pickle.load(open(path_infile, "rb"))
    # print("data", data)
    # global tag
    global_end = data["metadata"]["last_bar"] * BAR_RESOL
    instr_idxs = data["notes"].keys()

    ## === MUSICAL CONTENT ===
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_event("Bar", None))

        if order_first == "beat":
            bar_events, instruments_pitch_range, chords_in_bar = (
                make_sequence_one_bar_beat_first(
                    data, bar_step, instr_idxs, multi_track, pitch_encoding
                )
            )
        elif order_first == "track":
            bar_events, instruments_pitch_range, chords_in_bar = (
                make_sequence_one_bar_track_first(
                    data, bar_step, instr_idxs, multi_track, pitch_encoding
                )
            )

        ## === HEADER ===
        # NTracks in token
        # n_instr = min(MAX_INSTRUMENTS, len(instruments_in_bar))
        # final_sequence.append(create_event("Description_CountTracks", n_instr))

        # Instrument block
        if multi_track:
            final_sequence.append(create_event("Description_BeginTracks", None))
            for instr, pitches in instruments_pitch_range.items():
                if len(pitches) > 0:
                    final_sequence.append(create_event("Description_Track", instr))
            final_sequence.append(create_event("Description_EndTracks", None))

        if multi_track and header_pitch_range:
            # Pitch range block
            final_sequence.append(create_event("Description_BeginPitchRange", None))
            for instr, pitches in instruments_pitch_range.items():
                if len(pitches) > 0:
                    avg_pitch = min(130, roundup(np.mean(pitches)))
                    final_sequence.append(
                        create_event("Description_PitchRange", f"{instr}-{avg_pitch}")
                    )
            final_sequence.append(create_event("Description_EndPitchRange", None))

        if multi_track and header_repeatability:
            final_sequence.append(create_event("Description_BeginRepeatability", None))
            for instr, pitches in instruments_pitch_range.items():
                different_pitches = min(12, len(set(pitches)))
                final_sequence.append(
                    create_event(
                        "Description_Repeatability", f"{instr}-{different_pitches}"
                    )
                )
            final_sequence.append(create_event("Description_EndRepeatability", None))

        # Chord block
        if header_chords:
            final_sequence.append(create_event("Description_BeginChords", None))
            for chord in chords_in_bar:
                final_sequence.append(create_event("Description_Chord", chord))
            final_sequence.append(create_event("Description_EndChords", None))

        final_sequence.extend(bar_events)

    # BAR ending
    final_sequence.append(create_event("Bar", None))

    # EOS
    final_sequence.append(create_event("EOS", None))

    ## === POST-PROCESSING ===
    # Remove duplicate notes
    if not (multi_track):
        new_final_sequence = []
        already_present = []
        cur_bar, cur_beat = 0, 0
        i = 0
        while i < len(final_sequence):
            if (
                pitch_encoding == "absolute"
                and final_sequence[i]["name"] == "Note_Pitch"
                and (i + 1) < len(final_sequence)
                and final_sequence[i + 1]["name"] == "Note_Velocity"
                and (i + 2) < len(final_sequence)
                and final_sequence[i + 2]["name"] == "Note_Duration"
            ):
                pitch = final_sequence[i]["value"]
                duration = final_sequence[i + 2]["value"]
                if (cur_bar, cur_beat, pitch, duration) not in already_present:
                    already_present.append((cur_bar, cur_beat, pitch, duration))
                    new_final_sequence.extend(
                        [
                            final_sequence[i],
                            final_sequence[i + 1],
                            final_sequence[i + 2],
                        ]
                    )
                i += 3
            elif (
                pitch_encoding == "pitchclass"
                and final_sequence[i]["name"] == "Note_PitchClass"
                and (i + 1) < len(final_sequence)
                and final_sequence[i + 1]["name"] == "Note_Octave"
                and (i + 2) < len(final_sequence)
                and final_sequence[i + 2]["name"] == "Note_Velocity"
                and (i + 3) < len(final_sequence)
                and final_sequence[i + 3]["name"] == "Note_Duration"
            ):
                pitch_class = final_sequence[i]["value"]
                octave = final_sequence[i + 1]["value"]
                duration = final_sequence[i + 2]["value"]
                if (
                    cur_bar,
                    cur_beat,
                    pitch_class,
                    octave,
                    duration,
                ) not in already_present:
                    already_present.append(
                        (cur_bar, cur_beat, pitch_class, octave, duration)
                    )
                    new_final_sequence.extend(
                        [
                            final_sequence[i],
                            final_sequence[i + 1],
                            final_sequence[i + 2],
                            final_sequence[i + 3],
                        ]
                    )
                i += 4
            else:
                new_final_sequence.append(final_sequence[i])
                if final_sequence[i]["name"] == "Bar":
                    cur_bar += 1
                elif final_sequence[i]["name"] == "Beat":
                    cur_beat = int(final_sequence[i]["value"])
                i += 1
        final_sequence = new_final_sequence

    ## === MELODY ===
    melody_mask = None
    melodic_tracks = None
    if melodic_tokens is not None:
        ## Compute the melody: is it beat-wise skyline or track-wise skyline?
        if melodic_tokens == "note" and order_first == "track":
            raise NotImplementedError(
                "Not implemented for note-melodic-tokens and order-track-first"
            )

        if melodic_tokens == "note":
            list_melodic_tokens, indexes_melody_start, indexes_melody_end = (
                extract_melody_events(final_sequence)
            )
        elif melodic_tokens == "track":
            indexes_melody_start, indexes_melody_end, melodic_tracks = (
                extract_melody_track(final_sequence, order_first=order_first)
            )
        else:
            raise AttributeError(
                "Wrong melodic tokens provided: {}".format(melodic_tokens)
            )

        final_sequence_melody = []
        melody_mask = np.zeros(len(final_sequence), dtype=int)
        is_in_melody = False
        for ii, tok in enumerate(final_sequence):
            if ii in indexes_melody_start:
                final_sequence_melody.append(create_event("MelodyBegin", None))
                is_in_melody = True
            if ii in indexes_melody_end:
                final_sequence_melody.append(create_event("MelodyEnd", None))
                is_in_melody = False
            if tok["name"] in ["Bar", "Beat"] or is_in_melody:
                melody_mask[ii] = 1
            elif tok["name"] == "Chord" and INCLUDE_CHORDS_IN_MELODY:
                melody_mask[ii] = 1

            final_sequence_melody.append(tok)

        ## Where is the melody written: In the sequence or in the header?
        if melodic_tokens_position == "note":
            final_sequence = final_sequence_melody
        elif melodic_tokens_position == "before":
            melody_events = make_melodic_sequence(
                final_sequence_melody, include_chords=INCLUDE_CHORDS_IN_MELODY
            )
            assert (
                list(itertools.compress(final_sequence, melody_mask)) == melody_events
            )
            melody_bar_pos = []
            for idx, event in enumerate(melody_events):
                if event["name"] == "Bar":
                    melody_bar_pos.append(idx)

        ## Add information about the melodic instrument in the header?
        if header_melodic_instr:
            assert melodic_tracks is not None, "Melodic information is missing"
            final_sequence_with_mel_instr = []
            idx_bar = -1
            for ii, event in enumerate(final_sequence):
                if event["name"] == "Bar":
                    idx_bar += 1
                elif event["name"] == "Description_EndTracks":
                    final_sequence_with_mel_instr.append(
                        create_event("Description_BeginMelInstr", None)
                    )
                    for instr in melodic_tracks[idx_bar]:
                        final_sequence_with_mel_instr.append(
                            create_event("Description_MelInstr", instr)
                        )
                    final_sequence_with_mel_instr.append(
                        create_event("Description_EndMelInstr", None)
                    )
                final_sequence_with_mel_instr.append(event)
            final_sequence = final_sequence_with_mel_instr

    ## === SAVE ===
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[: -len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, "wb"))

    # save melodic mask
    if path_maskfile is not None and melody_mask is not None:
        fn = os.path.basename(path_maskfile)
        os.makedirs(path_maskfile[: -len(fn)], exist_ok=True)
        pickle.dump(
            {
                "melody_bar_pos": melody_bar_pos,
                "mask": melody_mask,
                "melody_events": melody_events,
                "melodic_tracks": melodic_tracks,
            },
            open(path_maskfile, "wb"),
        )

    return len(final_sequence)


# ==========================================================================================================
# ==========================================================================================================


def main(
    path_indir: str = "./corpus",
    path_outdir: str = os.path.join(path_root_default, "events"),
    path_maskdir: Optional[str] = None,
    tokenization: str = "remi",
    multi_track: bool = True,
    pitch_encoding: str = "absolute",
    header_pitch_range: bool = True,
    header_repeatability: bool = True,
    header_chords: bool = True,
    header_melodic_instr: bool = False,
    melodic_tokens_position: Optional[str] = None,
    melodic_tokens: Optional[str] = None,
    order_first: str = "beat",
    VERBOSE: bool = True,
):
    """Convert corpus files to event files sequentially

    Parameters
    ----------
    path_indir : str, optional
        Input directo, by default "./corpus"
    path_outdir : str, optional
        Output directory, by default os.path.join(path_root_default, "events")
    path_maskdir : Optional[str], optional
        Output directory for melodic mask, by default None
    tokenization : str, optional
        Tokenization strategy, by default "remi"
    multi_track : bool, optional
        Include track tokens, by default True
    pitch_encoding : str, optional
        Pitch encoding (absolute or pitchclass), by default "absolute"
    header_pitch_range : bool, optional
        Add pitch_range tokens in the header, by default True
    header_repeatability : bool, optional
        Add reapeatability tokens in the header, by default True
    header_chords : bool, optional
        Add chords tokens in the header, by default True
    header_melodic_instr : bool, optional
        Add melodic_instrument tokens in the header, by default False
    melodic_tokens_position : Optional[str], optional
        Where is the information of melodic events situated (before (ie. in the header) or note), by default None
    melodic_tokens : Optional[str], optional
        Skyline algorithm: (note-wise (ie beat-wise) or track-wise), by default None
    order_first : str, optional
        Beat-first or track-first parsing, by default "beat"
    VERBOSE : bool, optional
        Show output, by default True
    """
    # paths
    if path_maskdir is None:
        print("WARNING! NO MELODY TOKENS SAVED!")
    if path_maskdir is not None:
        os.makedirs(path_maskdir, exist_ok=True)
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(path_indir, extension=("pkl"), is_pure=True, is_sort=True)
    n_files = len(midifiles)
    if VERBOSE:
        print("num files:", n_files)

    # run all
    len_list = []
    for fidx in tqdm(range(n_files), leave=False):
        path_midi = midifiles[fidx]
        # print("{}/{}".format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)
        path_maskfile = (
            os.path.join(path_maskdir, path_midi) if path_maskdir is not None else None
        )

        # proc
        num_tokens = corpus2event_remi_v2(
            path_infile,
            path_outfile,
            path_maskfile=path_maskfile,
            multi_track=multi_track,
            header_pitch_range=header_pitch_range,
            header_repeatability=header_repeatability,
            header_chords=header_chords,
            header_melodic_instr=header_melodic_instr,
            pitch_encoding=pitch_encoding,
            melodic_tokens=melodic_tokens,
            melodic_tokens_position=melodic_tokens_position,
            order_first=order_first,
        )

        # print(" > num_token:", num_tokens)
        len_list.append(num_tokens)

    # plot
    # plot_hist(len_list, os.path.join(path_root_default, "num_tokens.png"))


def main_parallel(
    path_indir: str = "./corpus",
    path_outdir: str = os.path.join(path_root_default, "events"),
    path_maskdir: Optional[str] = None,
    tokenization: str = "remi",
    multi_track: bool = True,
    pitch_encoding: str = "absolute",
    header_pitch_range: bool = True,
    header_repeatability: bool = True,
    header_chords: bool = True,
    header_melodic_instr: bool = False,
    melodic_tokens_position: Optional[str] = None,
    melodic_tokens: Optional[str] = None,
    order_first: str = "beat",
):
    """Convert corpus files to event files in parallel

    Parameters
    ----------
    path_indir : str, optional
        Input directo, by default "./corpus"
    path_outdir : str, optional
        Output directory, by default os.path.join(path_root_default, "events")
    path_maskdir : Optional[str], optional
        Output directory for melodic mask, by default None
    tokenization : str, optional
        Tokenization strategy, by default "remi"
    multi_track : bool, optional
        Include track tokens, by default True
    pitch_encoding : str, optional
        Pitch encoding (absolute or pitchclass), by default "absolute"
    header_pitch_range : bool, optional
        Add pitch_range tokens in the header, by default True
    header_repeatability : bool, optional
        Add reapeatability tokens in the header, by default True
    header_chords : bool, optional
        Add chords tokens in the header, by default True
    header_melodic_instr : bool, optional
        Add melodic_instrument tokens in the header, by default False
    melodic_tokens_position : Optional[str], optional
        Where is the information of melodic events situated (before (ie. in the header) or note), by default None
    melodic_tokens : Optional[str], optional
        Skyline algorithm: (note-wise (ie beat-wise) or track-wise), by default None
    order_first : str, optional
        Beat-first or track-first parsing, by default "beat"
    """
    # paths
    if path_maskdir is not None:
        os.makedirs(path_maskdir, exist_ok=True)
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(path_indir, extension=("pkl"), is_pure=True, is_sort=True)
    n_files = len(midifiles)
    print("num files:", n_files)

    # run all
    list_data = []
    for fidx in tqdm(range(n_files), leave=False):
        path_midi = midifiles[fidx]
        # print("{}/{}".format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)
        path_maskfile = (
            os.path.join(path_maskdir, path_midi) if path_maskdir is not None else None
        )

        # proc
        list_data.append(
            {
                "path_infile": path_infile,
                "path_outfile": path_outfile,
                "path_maskfile": path_maskfile,
            }
        )

    df_proc = pd.DataFrame(list_data)

    parallel_proc_one = lambda row: corpus2event_remi_v2(
        row.path_infile,
        row.path_outfile,
        path_maskfile=row.path_maskfile,
        multi_track=multi_track,
        header_pitch_range=header_pitch_range,
        header_repeatability=header_repeatability,
        header_melodic_instr=header_melodic_instr,
        header_chords=header_chords,
        pitch_encoding=pitch_encoding,
        melodic_tokens=melodic_tokens,
        melodic_tokens_position=melodic_tokens_position,
        order_first=order_first,
    )

    df_proc.parallel_apply(parallel_proc_one, axis=1) # type: ignore


if __name__ == "__main__":
    main()
