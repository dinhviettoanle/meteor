import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import yaml
from scipy.stats import entropy
import chorder
import miditoolkit

from .dataloader import REMIFullSongTransformerDataset
from .remi2midi import remi2midi
from .utils import pickle_load
from .custom_data.melody_extraction import *

temp_dir = "temp5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+")
    args = parser.parse_args()

    config_path = args.config[0]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    config = None

if config is not None:
    device = config["training"]["device"]
    data_dir = config["data"]["data_dir"]
    vocab_path = config["data"]["vocab_path"]
    data_split = config["data"]["test_split"]
    multitrack_mapping = config["data"].get("multitrack_mapping")
    fine_control = config["generate"].get("fine_control", {})
    tconfig = config.get("tokenization")

    out_dir = args.config[1]
    n_pieces = int(args.config[2])
    n_samples_per_piece = int(args.config[3])

    pprint(config)

INSTR_TO_REGISTER = json.load(open("custom_data/instr_to_register.json", "r"))
INSTR_TO_REPEATABILITY = json.load(open("custom_data/instr_to_repeatability.json", "r"))

if (config is not None) and (config["generate"].get("gen_from_midi")):
    from .attributes import (
        analyze_local_attributes as attributes_analyze_local_attributes,
    )  # type: ignore
    from .attributes import main as attributes_main
    from .attributes import proc_one as attributes_proc_one
    from .custom_data.analyzer import main as analyzer_main
    from .custom_data.corpus2events import main as corpus2events_main
    from .custom_data.events2musemorphose import main as events2musemorphose_main
    from .custom_data.midi2corpus import main as midi2corpus_main


###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def get_beat_idx(event):
    return int(event.split("_")[-1])


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def extract_melody(events):
    melody_events = []
    notes_in_beat = []
    is_in_melody = False

    for ii, ev in enumerate(events):
        # if ev['name'] == 'Track':
        #     continue
        if is_in_melody:
            melody_events.append(ev)

        if "Bar" in ev or "Beat" in ev:
            melody_events.append(ev)
        elif "MelodyBegin" in ev:
            is_in_melody = True
        elif "MelodyEnd" in ev:
            is_in_melody = False

    return melody_events


#############################################################


def a_posteriori_analysis(midi_file, folder_analysis="temp_analysis", temporary=True):
    assert config is not None, "Config file is not set"
    print("[analysis] Analyzing generated piece:", midi_file)

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
        multi_track=config["data"].get("multi_track", True),
        multi_track_mapping=config["data"].get("multitrack_mapping"),
        VERBOSE=False,
    )

    corpus2events_main(
        path_indir=os.path.join(folder_analysis, "corpus"),
        path_outdir=os.path.join(folder_analysis, "events"),
        path_maskdir=os.path.join(folder_analysis, "melody"),
        multi_track=config["data"].get("multi_track", True),
        pitch_encoding=tconfig["pitch_encoding"],
        header_chords=tconfig["header_chords"],
        header_repeatability=tconfig["header_repeatability"],
        header_pitch_range=tconfig["header_pitch_range"],
        melodic_tokens=tconfig["melodic_tokens"],
        melodic_tokens_position=tconfig["melodic_tokens_position"],
        order_first=tconfig["order_first"],
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
    print(f"[analysis] rhythm {computed_attributes['rfreq_cls']}")
    print(f"[analysis] polyph {computed_attributes['polyph_cls']}")
    print(f"[analysis] harmdiv {computed_attributes['harmdiv_cls']}")
    print(f"[analysis] rhythmdiv {computed_attributes['rhythmdiv_cls']}")

    local_attributes = attributes_analyze_local_attributes(
        "1.pkl", data_dir=os.path.join(folder_analysis, "events")
    )

    if temporary:
        shutil.rmtree(folder_analysis)

    return {
        "rfreq_cls": computed_attributes["rfreq_cls"],
        "polyph_cls": computed_attributes["polyph_cls"],
        "harmdiv_cls": computed_attributes["harmdiv_cls"],
        "rhythmdiv_cls": computed_attributes["rhythmdiv_cls"],
        "local_attributes": local_attributes,
    }


def analyze_from_midi(midi_path):
    print("[info] Analyzing custom MIDI file...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    os.makedirs(os.path.join(temp_dir, "midi_raw"))
    shutil.copy2(midi_path, os.path.join(temp_dir, "midi_raw"))

    a_posteriori_analysis(None, folder_analysis=temp_dir, temporary=False)

    attributes_main(
        data_dir=os.path.join(temp_dir, "remi"),
        polyph_out_dir=os.path.join(temp_dir, "remi/attr_cls/polyph"),
        rhythm_out_dir=os.path.join(temp_dir, "remi/attr_cls/rhythm"),
        harmdiv_out_dir=os.path.join(temp_dir, "remi/attr_cls/harmdiv"),
        rhythmdiv_out_dir=os.path.join(temp_dir, "remi/attr_cls/rhythmdiv"),
        rhythm_local_out_dir=os.path.join(temp_dir, "remi/attr_cls/local_rhythm"),
        pitch_local_out_dir=os.path.join(temp_dir, "remi/attr_cls/local_pitch"),
        config_out_path=os.path.join(temp_dir, "remi/attr_cls"),
    )

    return temp_dir


#############################################################
#############################################################


if __name__ == "__main__":
    if config is None: 
        raise FileNotFoundError("Config file has not been set")

    mconf = config["model"]

    ## From what we generate
    if config["generate"].get("gen_from_midi"):
        temp_dir = analyze_from_midi(config["generate"]["gen_from_midi"])
        data_dir = os.path.join(temp_dir, "remi")
        if config["data"].get("melody_mask_dir"):
            config["data"]["melody_mask_dir"] = os.path.join(temp_dir, "melody")
        config["generate"]["gen_from_piece"] = "1"

    ## Melody instruments
    melody_instruments = None
    return_melody_tokens = False
    if config["generate"].get("fine_control") and config["generate"][
        "fine_control"
    ].get("melody_instruments"):
        melody_instruments = config["generate"]["fine_control"].get(
            "melody_instruments"
        )
        return_melody_tokens = True

    return_melody_tokens = True

    dset = REMIFullSongTransformerDataset(
        data_dir,
        vocab_path,
        do_augment=False,
        model_enc_seqlen=config["data"]["enc_seqlen"],
        model_dec_seqlen=config["generate"]["dec_seqlen"],
        model_max_bars=config["generate"]["max_bars"],
        pieces=(
            pickle_load(data_split)
            if not (config["generate"].get("gen_from_midi"))
            else ["1.pkl"]
        ),
        use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
        pad_to_same=False,
        files_limit=config["data"].get("files_limit", None),
        gen_from_piece=config["generate"].get("gen_from_piece"),
        gen_from_bar=config["generate"].get("gen_from_bar"),
        return_melody_tokens=return_melody_tokens,
        melody_first=config["data"].get("melody_first", False),
        melody_mask_dir=config["data"].get("melody_mask_dir", None),
        force_return_melody=config["data"].get("force_return_melody", False),
    )

    if config["generate"].get("pieces") is not None:
        random.seed(config["generate"]["seed"])
    else:
        pieces = random.sample(range(len(dset)), n_pieces)
    print("[sampled pieces]", pieces)

    # if os.path.exists(out_dir):
    #   shutil.rmtree(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("============= GENERATING =============")

    times = []
    for idx_p, p in enumerate(pieces):
        print("[info] piece {} / {}".format(idx_p, len(pieces)))

        # fetch test sample
        p_data = dset[p]
        p_id = p_data["piece_id"]
        p_bar_id = p_data["st_bar_id"]
        p_melody_tokens = p_data["melody_tokens_only"]
        p_data["enc_input"] = p_data["enc_input"][: p_data["enc_n_bars"]]
        p_data["enc_padding_mask"] = p_data["enc_padding_mask"][: p_data["enc_n_bars"]]

        orig_p_cls_str = "".join(str(c) for c in p_data["polyph_cls_bar"])
        orig_r_cls_str = "".join(str(c) for c in p_data["rhymfreq_cls_bar"])

        orig_song = p_data["dec_input"].tolist()[: p_data["length"]]
        orig_song = word2event(orig_song, dset.idx2event)
        piece_name = "id{}_bar{}_orig".format(p_id, p_bar_id)
        orig_out_file = os.path.join(out_dir, piece_name)
        print("[info] writing to ...", orig_out_file)
        # output reference song's MIDI
        os.makedirs(orig_out_file, exist_ok=True)
        initial_midi_obj, initial_chords = remi2midi(
            orig_song,
            os.path.join(orig_out_file, piece_name + ".mid"),
            return_chords=True,
            enforce_tempo=False,
            multitrack_mapping=multitrack_mapping,
        )

        # save metadata of reference song (events & attr classes)
        print(
            *orig_song,
            sep="\n",
            file=open(os.path.join(orig_out_file, piece_name + ".txt"), "a"),
        )

        melody_song = word2event(p_data["melody_tokens_list"], dset.idx2event)
        leadsheet_midi_obj, mel_chords = remi2midi(
            melody_song,
            output_midi_path=None,
            return_chords=True,
            enforce_tempo=False,
            multitrack_mapping=multitrack_mapping,
        )

        max_tick = 0
        for instr in leadsheet_midi_obj.instruments:
            for n in instr.notes:
                max_tick = max(n.start + n.duration, max_tick)

        mel_chords = [x for x in mel_chords if x.chord_val != "N_N"]

        harmony_instr = miditoolkit.Instrument(program=0, is_drum=False, name="chords")
        for ii, chord in enumerate(mel_chords):
            chord_symbol = chord.chord_val.replace("Chord-", "").replace("_", "")
            print(chord_symbol, end=" ")
            chorder_chord = chorder.Chord(chord_symbol)
            next_chord_start = (
                max_tick if ii == len(mel_chords) - 1 else mel_chords[ii + 1].start_tick
            )
            for pitch in chorder.chord_to_midi(chorder_chord): # type: ignore
                harmony_instr.notes.append(
                    miditoolkit.Note(
                        100, pitch, int(chord.start_tick), next_chord_start
                    )
                )

        leadsheet_midi_obj.instruments.append(harmony_instr)
        leadsheet_midi_obj.dump(os.path.join(orig_out_file, "lead sheet.mid"))
        print()
