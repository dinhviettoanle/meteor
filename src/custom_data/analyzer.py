import copy
import multiprocessing as mp
import os

import miditoolkit
import numpy as np
from chorder import Dechorder
from miditoolkit.midi.containers import Marker
from .utils import traverse_dir

# path_indir_default = "./midi_synchronized"
path_outdir_default = "./midi_analyzed"

path_indir_default = "./SymphonyNet_Dataset/classical"
path_outdir_default = "./symphonynet_classical_analyzed"

num2pitch = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


def proc_one(path_infile, path_outfile, VERBOSE=True):
    if VERBOSE:
        print("----")
        print(" >", path_infile)
        print(" >", path_outfile)

    # load
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile) # type: ignore
    midi_obj_out = copy.deepcopy(midi_obj)
    notes = midi_obj.instruments[0].notes
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # --- chord --- #
    # exctract chord
    chords = Dechorder.dechord(midi_obj)
    markers = []
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = (
                num2pitch[chord.root_pc]
                + "_"
                + chord.quality
                + "_"
                + num2pitch[chord.bass_pc]
            )
        else:
            chord_text = "N_N_N"
        markers.append(Marker(time=int(cidx * 480), text=chord_text))

    # dedup
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)

    # --- global properties --- #
    # global tempo
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm = int(tempo_median)
    if VERBOSE:
        print(" > [global] bpm:", global_bpm)

    # === save === #
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[: -len(fn)], exist_ok=True)

    # markers
    midi_obj_out.markers = dedup_chords
    midi_obj_out.markers.insert(
        0, Marker(text="global_bpm_" + str(int(global_bpm)), time=0)
    )

    # save
    # midi_obj_out.instruments[0].name = "piano"
    midi_obj_out.dump(path_outfile)
    if VERBOSE:
        print("Done")


def try_catch_proc_one(path_indir, path_outfile, VERBOSE=True):
    try:
        proc_one(path_indir, path_outfile, VERBOSE=VERBOSE)
    except Exception:
        print("Error for", path_indir)


def main(path_indir: str = path_indir_default, path_outdir: str = path_outdir_default, VERBOSE: bool = True):
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(path_indir, is_pure=True, is_sort=True)
    n_files = len(midifiles)
    if VERBOSE:
        print("num fiels:", n_files)

    # collect
    data = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        if VERBOSE:
            print("{}/{}".format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)

        # append
        data.append([path_infile, path_outfile, VERBOSE])

    # run, multi-thread
    pool = mp.Pool(processes=8)
    # pool.starmap(proc_one, data)
    pool.starmap(try_catch_proc_one, data)


if __name__ == "__main__":
    main()
