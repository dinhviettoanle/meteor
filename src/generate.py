import argparse
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from pprint import pprint
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from scipy.stats import entropy

from .dataloader import REMIFullSongTransformerDataset
from .model.musemorphose import MuseMorphose
from .remi2midi import remi2midi, copy_paste_track
from .utils import numpy_to_tensor, pickle_load, tensor_to_numpy

temp_dir = "temp"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+")
    parser.add_argument("--analyze", action="store_true", default=False)
    parser.add_argument("--random_pitchrange", action="store_true", default=False)
    parser.add_argument("--random_repeatability", action="store_true", default=False)
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

    ckpt_path = args.config[1]
    out_dir = args.config[2]
    n_pieces = int(args.config[3])
    n_samples_per_piece = int(args.config[4])
    with_analysis = args.analyze
    print("[info] with a posteriori analysis:", with_analysis)

    with_random_pitchrange = args.random_pitchrange
    print("[info] with random pitchrange:", with_random_pitchrange)

    with_random_repeatability = args.random_repeatability
    print("[info] with random repeatability:", with_random_repeatability)

    pprint(config)

    if with_analysis or config["generate"].get("gen_from_midi"):
        from .attributes import (
            analyze_local_attributes as attributes_analyze_local_attributes,
        )  # type: ignore
        from .attributes import main as attributes_main
        from .attributes import proc_one as attributes_proc_one
        from .custom_data.analyzer import main as analyzer_main
        from .custom_data.corpus2events import main as corpus2events_main
        from .custom_data.events2musemorphose import main as events2musemorphose_main
        from .custom_data.midi2corpus import main as midi2corpus_main


custom_midi_config = {
    "multi_track": True,
    "multi_track_mapping": "mmt",
    "tokenization": "remi",
    "pitch_encoding": "absolute",
}

INSTR_TO_REGISTER = json.load(open("custom_data/instr_to_register.json", "r"))
INSTR_TO_REPEATABILITY = json.load(open("custom_data/instr_to_repeatability.json", "r"))


###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def get_beat_idx(event):
    return int(event.split("_")[-1])


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print("overflow detected, use 128-bit")
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


########################################
# generation
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0.0):
    # reshape
    batch_inp = piece_data["enc_input"].permute(1, 0).long().to(device)
    batch_padding_mask = piece_data["enc_padding_mask"].bool().to(device)

    # get latent conditioning vectors
    with torch.no_grad():
        piece_latents = model.get_sampled_latent(
            batch_inp,
            padding_mask=batch_padding_mask,
            use_sampling=use_sampling,
            sampling_var=sampling_var,
        )

    return piece_latents


def make_prime_bar_instr(instruments_priming_in_bar, with_bar=False):
    instruments = [x.get("instrument") for x in instruments_priming_in_bar]
    pitchranges = [x.get("pitchrange") for x in instruments_priming_in_bar]
    repeatabilities = [x.get("repeatability") for x in instruments_priming_in_bar]
    primer = []
    if with_bar:
        primer.append("Bar_None")

    # Count Tracks
    # primer += [f'Description_CountTracks_{len(instruments_priming_in_bar)}'] + [f'Description_Track_{inst}' for inst in instruments_priming_in_bar]

    # Instrument blocks
    if None not in instruments:
        primer += (
            ["Description_BeginTracks_None"]
            + [f"Description_Track_{inst}" for inst in instruments]
            + ["Description_EndTracks_None"]
        )
    if None not in pitchranges:
        primer += (
            ["Description_BeginPitchRange_None"]
            + [
                f"Description_PitchRange_{inst}-{value}"
                for inst, value in zip(instruments, pitchranges)
            ]
            + ["Description_EndPitchRange_None"]
        )
    if None not in repeatabilities:
        primer += (
            ["Description_BeginRepeatability_None"]
            + [
                f"Description_Repeatability_{inst}-{value}"
                for inst, value in zip(instruments, repeatabilities)
            ]
            + ["Description_EndRepeatability_None"]
        )

    return primer



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


def generate_on_latent_ctrl_vanilla_truncate(
    model,
    latents,
    rfreq_cls,
    polyph_cls,
    harmdiv_cls,
    rhythmdiv_cls,
    event2idx,
    idx2event,
    max_events=8000,
    primer=None,
    max_input_len=1280,
    truncate_len=512,
    nucleus_p=0.9,
    temperature=1.2,
    use_attr_cls=True,
    use_attr_multitrack_cls=True,
    instruments_priming=None,
    melody_tokens: Optional[Dict] = None,
    melody_instruments=None,
    pitchrange_diff=None,
    repeatability_diff=None,
    infer_octave=True,
):
    latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
    rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device) # type: ignore
    polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device) # type: ignore
    harmdiv_placeholder = torch.zeros(max_events, 1, dtype=int).to(device) # type: ignore
    rhythmdiv_placeholder = torch.zeros(max_events, 1, dtype=int).to(device) # type: ignore
    print(
        "[info] rhythm cls: {} | polyph_cls: {}".format(
            rfreq_cls.tolist(), polyph_cls.tolist()
        )
    )
    print(
        "[info] rhythm harmdiv_cls: {} | rhythmdiv_cls: {}".format(
            harmdiv_cls.tolist(), rhythmdiv_cls.tolist()
        )
    )

    if primer is None:
        generated = [event2idx["Bar_None"]]
    else:
        generated = [event2idx[e] for e in primer]
        latent_placeholder[: len(generated), 0, :] = latents[0].squeeze(0)
        rfreq_placeholder[: len(generated), 0] = rfreq_cls[0]
        polyph_placeholder[: len(generated), 0] = polyph_cls[0]
        harmdiv_placeholder[: len(generated), 0] = harmdiv_cls[0]
        rhythmdiv_placeholder[: len(generated), 0] = rhythmdiv_cls[0]

    target_bars, generated_bars = latents.size(0), 0

    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    cur_input_len = len(generated)
    generated_final = deepcopy(generated)
    entropies = []

    list_pitchranges = []
    list_repeatabilities = []
    bar_pitchranges = []
    bar_repeatabilities = []
    force_pitchclass = False
    remaining_melodic_tokens = []
    tracks_in_bar = []
    all_probs = []

    while generated_bars < target_bars:
        print(cur_input_len, end="\r")

        if len(generated) == 1:
            dec_input = numpy_to_tensor([generated], device=device).long()
        else:
            dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

        latent_placeholder[len(generated) - 1, 0, :] = latents[generated_bars]
        rfreq_placeholder[len(generated) - 1, 0] = rfreq_cls[generated_bars]
        polyph_placeholder[len(generated) - 1, 0] = polyph_cls[generated_bars]
        if use_attr_multitrack_cls:
            harmdiv_placeholder[len(generated) - 1, 0] = harmdiv_cls[generated_bars]
            rhythmdiv_placeholder[len(generated) - 1, 0] = rhythmdiv_cls[generated_bars]

        dec_seg_emb = latent_placeholder[: len(generated), :]
        if use_attr_cls:
            dec_rfreq_cls = rfreq_placeholder[: len(generated), :]
            dec_polyph_cls = polyph_placeholder[: len(generated), :]
        else:
            dec_rfreq_cls = None
            dec_polyph_cls = None

        if use_attr_multitrack_cls:
            dec_harmdiv_cls = harmdiv_placeholder[: len(generated), :]
            dec_rhythmdiv_cls = rhythmdiv_placeholder[: len(generated), :]
        else:
            dec_harmdiv_cls = None
            dec_rhythmdiv_cls = None

        # sampling
        with torch.no_grad():
            logits = model.generate(
                dec_input,
                dec_seg_emb,
                dec_rfreq_cls,
                dec_polyph_cls,
                dec_harmdiv_cls,
                dec_rhythmdiv_cls,
            )
        logits = tensor_to_numpy(logits[0])
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)
        word_event = idx2event[word]
        all_probs.append(probs[word])

        if "Beat" in word_event:
            event_pos = get_beat_idx(word_event)
            print("Beat {}".format(event_pos))
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print("[info] position not increasing, failed cnt:", failed_cnt)
                if failed_cnt >= 128:
                    print("[FATAL] model stuck, exiting ...")
                    return (
                        False,
                        generated,
                        time.time() - time_st,
                        np.array(entropies),
                        np.array(all_probs),
                        None,
                    )
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        if "Bar" in word_event:
            generated_bars += 1
            cur_pos = 0
            print(
                "[info] generated {} bars, #events = {}".format(
                    generated_bars, len(generated_final)
                )
            )

        if word_event == "PAD_None":
            continue

        if "Description_Track" in word_event:
            tracks_in_bar.append(word_event.replace("Description_Track_", ""))

        if len(generated) > max_events or (
            word_event == "EOS_None" and generated_bars == target_bars - 1
        ):
            generated_bars += 1
            generated.append(event2idx["Bar_None"])
            print("[info] gotten eos")
            break

        generated.append(word)
        generated_final.append(word)
        entropies.append(entropy(probs))

        cur_input_len += 1
        steps += 1

        # Track-level conditions
        if pitchrange_diff is not None and "Description_BeginPitchRange" in word_event:
            pitchrange_events = []
            for instr in tracks_in_bar:
                new_value = roundup(INSTR_TO_REGISTER[instr]["mean"] + pitchrange_diff)
                new_pitchrange = "Description_PitchRange_{}-{}".format(instr, new_value)
                if new_pitchrange not in pitchrange_events:
                    pitchrange_events.append(new_pitchrange)
                    bar_pitchranges.append(new_pitchrange)
            print(f"[condition] {pitchrange_events}")
            pitchrange_tokens = [event2idx[e] for e in pitchrange_events]
            cur_input_len += len(pitchrange_tokens)
            generated += pitchrange_tokens
            generated_final += pitchrange_tokens

        if (
            repeatability_diff is not None
            and "Description_BeginRepeatability" in word_event
        ):
            repeatability_events = []
            for instr in tracks_in_bar:
                new_value = np.clip(
                    int(INSTR_TO_REPEATABILITY[instr]["mean"]) + repeatability_diff,
                    1,
                    13,
                )
                new_repeatability = "Description_Repeatability_{}-{}".format(
                    instr, new_value
                )
                if new_repeatability not in repeatability_events:
                    repeatability_events.append(new_repeatability)
                    bar_repeatabilities.append(new_repeatability)
            print(f"[condition] {repeatability_events}")
            repeatability_tokens = [event2idx[e] for e in repeatability_events]
            cur_input_len += len(repeatability_tokens)
            generated += repeatability_tokens
            generated_final += repeatability_tokens

        # Melody forcing
        if "Beat" in word_event:
            if melody_tokens is not None and melody_instruments is not None:
                bar_query_melody_instrument = melody_instruments[generated_bars]
                bar_query_events = melody_tokens[generated_bars].get(event_pos)
                if bar_query_events and len(bar_query_events) > 0:
                    for instr in bar_query_melody_instrument:
                        if not (infer_octave):
                            mel_events = [f"Track_{instr}"] + bar_query_events
                        else:
                            mel_events = [f"Track_{instr}", bar_query_events[0]]
                            remaining_melodic_tokens = bar_query_events[2:]
                            force_pitchclass = True
                        print(instr, mel_events)
                        mel_tokens = [event2idx[e] for e in mel_events]
                        cur_input_len += len(mel_tokens)
                        generated += mel_tokens
                        generated_final += mel_tokens

        if force_pitchclass and "Octave" in word_event:
            print(
                "gen: {} | original: {} | remaining: {}".format(
                    word_event, bar_query_events[1], remaining_melodic_tokens
                )
            )
            mel_tokens = [event2idx[e] for e in remaining_melodic_tokens]
            cur_input_len += len(mel_tokens)
            generated += mel_tokens
            generated_final += mel_tokens
            force_pitchclass = False

        # Set priming for next bar
        if "Bar" in word_event:
            tracks_in_bar = []
            if instruments_priming is not None:
                bar_prime = make_prime_bar_instr(
                    instruments_priming[generated_bars], with_bar=False
                )
                bar_primer_idx = [event2idx[e] for e in bar_prime]
                print(bar_prime)
                cur_input_len += len(bar_primer_idx)
                generated += bar_primer_idx
                generated_final += bar_primer_idx
                tracks_in_bar = [
                    x.replace("Description_Track_", "") for x in bar_prime[1:-1]
                ]

        if "Bar" in word_event and cur_input_len > 0:
            list_pitchranges.append(bar_pitchranges)
            list_repeatabilities.append(bar_repeatabilities)
            bar_pitchranges = []
            bar_repeatabilities = []

        assert cur_input_len == len(generated)
        if cur_input_len == max_input_len:
            generated = generated[-truncate_len:]
            latent_placeholder[: len(generated) - 1, 0, :] = latent_placeholder[
                cur_input_len - truncate_len : cur_input_len - 1, 0, :
            ]
            rfreq_placeholder[: len(generated) - 1, 0] = rfreq_placeholder[
                cur_input_len - truncate_len : cur_input_len - 1, 0
            ]
            polyph_placeholder[: len(generated) - 1, 0] = polyph_placeholder[
                cur_input_len - truncate_len : cur_input_len - 1, 0
            ]
            harmdiv_placeholder[: len(generated) - 1, 0] = harmdiv_placeholder[
                cur_input_len - truncate_len : cur_input_len - 1, 0
            ]
            rhythmdiv_placeholder[: len(generated) - 1, 0] = rhythmdiv_placeholder[
                cur_input_len - truncate_len : cur_input_len - 1, 0
            ]

            print(
                "[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}. Last event: {}".format(
                    cur_input_len,
                    len(generated_final),
                    cur_input_len - truncate_len,
                    cur_input_len - 1,
                    word_event,
                )
            )
            cur_input_len = len(generated)
            if len(generated_final) > 10_000:
                print("[FATAL] too long...")
                return (
                    False,
                    generated,
                    time.time() - time_st,
                    np.array(entropies),
                    np.array(all_probs),
                    None,
                )

    assert generated_bars == target_bars
    print("-- generated events:", len(generated_final))
    print("-- time elapsed: {:.2f} secs".format(time.time() - time_st))

    local_conditions = {
        "pitchrange": list_pitchranges,
        "repeatability": list_repeatabilities,
    }
    # avg_p = sum([np.log(p) for p in all_probs]) / len(all_probs)
    # print(avg_p, np.exp(-avg_p))
    return (
        True,
        generated_final[:-1],
        time.time() - time_st,
        np.array(entropies),
        np.array(all_probs),
        local_conditions,
    )


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


########################################
# change attribute classes
########################################
def random_shift_attr_cls(n_samples, upper=4, lower=-4):
    return np.random.randint(lower, upper, (n_samples,))


def get_local_attributes_query(instruments_priming):
    if instruments_priming is None:
        return
    per_bar = []
    for instr_bar in instruments_priming:
        current_bar = {
            "pitchrange": [],
            "repeatability": [],
        }
        for instr in instr_bar:
            current_bar["pitchrange"].append(
                (instr["instrument"], instr.get("pitchrange", -1))
            )
            current_bar["repeatability"].append(
                (instr["instrument"], instr.get("repeatability", -1))
            )
        per_bar.append(current_bar)
    return per_bar


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

    ## Instrument priming
    instruments_priming = None
    if (
        config["generate"].get("use_primer")
        and config["generate"].get("fine_control")
        and config["generate"]["fine_control"].get("instruments")
    ):
        instruments_priming = config["generate"]["fine_control"].get("instruments")
        if isinstance(instruments_priming, dict) and instruments_priming.get("common"):
            print("[info] Common instrumentation provided")
            list_instruments_by_bar = [instruments_priming["common"]] * (
                config["generate"]["max_bars"] + 1
            )
            instruments_priming = list_instruments_by_bar

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

    # print(dset[3222]['piece_id'])
    # assert False

    if config["generate"].get("pieces") is not None:
        random.seed(config["generate"]["seed"])
    else:
        pieces = random.sample(range(len(dset)), n_pieces)
    print("[sampled pieces]", pieces)

    model = MuseMorphose(
        mconf["enc_n_layer"],
        mconf["enc_n_head"],
        mconf["enc_d_model"],
        mconf["enc_d_ff"],
        mconf["dec_n_layer"],
        mconf["dec_n_head"],
        mconf["dec_d_model"],
        mconf["dec_d_ff"],
        mconf["d_latent"],
        mconf["d_embed"],
        dset.vocab_size,
        d_polyph_emb=mconf["d_polyph_emb"],
        d_rfreq_emb=mconf["d_rfreq_emb"],
        d_harmdiv_emb=mconf["d_harmdiv_emb"],
        d_rhythmdiv_emb=mconf["d_rhythmdiv_emb"],
        cond_mode=mconf["cond_mode"],
        use_attr_cls=mconf["use_attr_cls"],
        use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
    ).to(device)
    model.eval()
    # load_state_dict(model, torch.load(ckpt_path, map_location='cpu'))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    print(
        "  [config] # parameters {:,}".format(
            sum(p.numel() for p in model.parameters())
        )
    )
    print("  [config] Use attr_cls", mconf["use_attr_cls"])
    print("  [config] Use attr_multitrack_cls", mconf["use_attr_multitrack_cls"])

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
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
        orig_out_file = os.path.join(out_dir, "id{}_bar{}_orig".format(p_id, p_bar_id))
        print("[info] writing to ...", orig_out_file)
        # output reference song's MIDI
        initial_midi_obj, orig_tempo = remi2midi(
            orig_song,
            orig_out_file + ".mid",
            return_first_tempo=True,
            enforce_tempo=False,
            multitrack_mapping=multitrack_mapping,
        )

        # save metadata of reference song (events & attr classes)
        print(*orig_song, sep="\n", file=open(orig_out_file + ".txt", "a"))
        np.save(orig_out_file + "-POLYCLS.npy", p_data["polyph_cls_bar"])
        np.save(orig_out_file + "-RHYMCLS.npy", p_data["rhymfreq_cls_bar"])

        print(
            "[info] initial rhythm cls: {} | polyph_cls: {}".format(
                torch.tensor(p_data["rhymfreq_cls_bar"]),
                torch.tensor(p_data["polyph_cls_bar"]),
            )
        )

        if with_analysis:
            try:
                a_posteriori_analysis(orig_out_file + ".mid", temporary=True)
            except Exception as e:
                print("Error", e)

        for k in p_data.keys():
            if isinstance(p_data[k], dict):
                continue
            if k == "melody_events":
                continue
            if not torch.is_tensor(p_data[k]):
                p_data[k] = numpy_to_tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)

        p_latents = get_latent_embedding_fast(
            model,
            p_data,
            use_sampling=config["generate"]["use_latent_sampling"],
            sampling_var=config["generate"]["latent_sampling_var"],
        )

        p_cls_diff = (
            np.array(fine_control.get("polyph_diff"))
            if fine_control.get("polyph_diff")
            else random_shift_attr_cls(n_samples_per_piece)
        )
        r_cls_diff = (
            np.array(fine_control.get("rfreq_diff"))
            if fine_control.get("rfreq_diff")
            else random_shift_attr_cls(n_samples_per_piece)
        )
        hdiv_cls_diff = (
            np.array(fine_control.get("harmdiv_diff"))
            if fine_control.get("harmdiv_diff")
            else random_shift_attr_cls(n_samples_per_piece)
        )
        rdiv_cls_diff = (
            np.array(fine_control.get("rhythmdiv_diff"))
            if fine_control.get("rhythmdiv_diff")
            else random_shift_attr_cls(n_samples_per_piece)
        )

        pitchrange_diff = (
            10 * random_shift_attr_cls(n_samples_per_piece)
            if with_random_pitchrange
            else None
        )
        repeatability_diff = (
            random_shift_attr_cls(n_samples_per_piece)
            if with_random_repeatability
            else None
        )
        if pitchrange_diff is not None:
            print("[info] pitchrange_diff: {}".format(pitchrange_diff))

        if config["generate"].get("use_primer"):
            if config["generate"]["fine_control"].get("instruments") and instruments_priming is not None:
                primer = make_prime_bar_instr(instruments_priming[0], with_bar=True)
            else:
                primer = []
                for ev in orig_song:
                    if (
                        "Bar" in ev
                        or "Description_Chord" in ev
                        or "Description_Track" in ev
                        or "BeginTracks" in ev
                        or "EndTracks" in ev
                    ):
                        primer.append(ev)
                    else:
                        break
        else:
            primer = None
        print("Primer:", primer)

        print()
        print("     ***** Generating conditioned samples *****")

        piece_entropies = []
        for samp in range(n_samples_per_piece):

            # user input cls
            p_polyph_cls = (
                torch.tensor(fine_control.get("polyph")).long()
                if fine_control.get("polyph")
                else (p_data["polyph_cls_bar"] + p_cls_diff[samp]).clamp(0, 7).long()
            )
            p_rfreq_cls = (
                torch.tensor(fine_control.get("rfreq")).long()
                if fine_control.get("rfreq")
                else (p_data["rhymfreq_cls_bar"] + r_cls_diff[samp]).clamp(0, 7).long()
            )
            p_harmdiv_cls = (
                torch.tensor(fine_control.get("harmdiv")).long()
                if fine_control.get("harmdiv")
                else (p_data["harmdiv_cls_bar"] + hdiv_cls_diff[samp])
                .clamp(0, 7)
                .long()
            )
            p_rhythmdiv_cls = (
                torch.tensor(fine_control.get("rhythmdiv")).long()
                if fine_control.get("rhythmdiv")
                else (p_data["rhythmdiv_cls_bar"] + rdiv_cls_diff[samp])
                .clamp(0, 7)
                .long()
            )

            print("[info] piece: {}, bar: {}".format(p_id, p_bar_id))
            out_file = os.path.join(
                out_dir,
                "id{}_bar{}_sample{:02d}_poly{}_rhym{}_hdiv{}_rdiv{}".format(
                    p_id,
                    p_bar_id,
                    samp + 1,
                    "".join(map(str, p_polyph_cls.tolist())),
                    "".join(map(str, p_rfreq_cls.tolist())),
                    "".join(map(str, p_harmdiv_cls.tolist())),
                    "".join(map(str, p_rhythmdiv_cls.tolist())),
                ),
            )
            print("[info] writing to ...", out_file)
            if os.path.exists(out_file + ".txt"):
                print("[info] file exists, skipping ...")
                continue

            # generate
            success, song, t_sec, entropies, all_probs, local_conditions = (
                generate_on_latent_ctrl_vanilla_truncate(
                    model,
                    p_latents,
                    p_rfreq_cls,
                    p_polyph_cls,
                    p_harmdiv_cls,
                    p_rhythmdiv_cls,
                    dset.event2idx,
                    dset.idx2event,
                    max_input_len=config["generate"]["max_input_dec_seqlen"],
                    truncate_len=min(
                        512, config["generate"]["max_input_dec_seqlen"] - 32
                    ),
                    nucleus_p=config["generate"]["nucleus_p"],
                    temperature=config["generate"]["temperature"],
                    use_attr_cls=mconf["use_attr_cls"],
                    use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
                    primer=primer,
                    instruments_priming=instruments_priming,
                    melody_tokens=p_melody_tokens,
                    melody_instruments=melody_instruments,
                    pitchrange_diff=(
                        pitchrange_diff[samp] if pitchrange_diff is not None else None
                    ),
                    repeatability_diff=(
                        repeatability_diff[samp]
                        if repeatability_diff is not None
                        else None
                    ),
                )
            )
            if not (success):
                print("Unsuccessful generation")
                continue

            times.append(t_sec)

            song = word2event(song, dset.idx2event)
            print(*song, sep="\n", file=open(out_file + ".txt", "a"))
            midi_obj = remi2midi(
                song,
                out_file + ".mid",
                enforce_tempo=True,
                enforce_tempo_val=orig_tempo,
                multitrack_mapping=multitrack_mapping,
            )
            copy_paste_track(
                out_file + "_melody.mid",
                initial_midi_obj=initial_midi_obj,
                generated_midi_obj=midi_obj,
                track_id=0,
            )

            # If using melody tokens
            # melody_events = extract_melody(song)
            # remi2midi(melody_events, is_full_event=False, output_midi_path=f'{out_file}_mel.mid', multitrack_mapping=multitrack_mapping)

            # Melody ground truth
            melody_events = p_data["melody_events"]
            if len(melody_events) > 0:
                remi2midi(
                    melody_events,
                    is_full_event=True,
                    output_midi_path=f"{out_file}_melonly.mid",
                    multitrack_mapping=multitrack_mapping,
                )

            if isinstance(p_melody_tokens, dict):
                with open(out_file + "_melody_tokens.json", "w") as fp:
                    json.dump(p_melody_tokens, fp)

            # save metadata of the generation
            # np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
            # np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))
            print(
                "[info] piece entropy: {:.4f} (+/- {:.4f})".format(
                    entropies.mean(), entropies.std()
                )
            )
            piece_entropies.append(entropies.mean())

            if with_analysis:
                try:
                    analysis_cls = a_posteriori_analysis(
                        out_file + ".mid", temporary=True
                    )
                    query_cls = {
                        "rfreq_cls": p_rfreq_cls.tolist(),
                        "polyph_cls": p_polyph_cls.tolist(),
                        "harmdiv_cls": p_harmdiv_cls.tolist(),
                        "rhythmdiv_cls": p_rhythmdiv_cls.tolist(),
                        "local_attributes": local_conditions,
                    }
                    with open(out_file + "_cls.json", "w") as fp:
                        json.dump(
                            {
                                "query": query_cls,
                                "analysis": analysis_cls,
                                "probs": all_probs.tolist(),
                                "entropy": entropies.tolist(),
                                "inference_time": t_sec,
                                "seq_len": len(song) if song is not None else None,
                            },
                            fp,
                            indent=2,
                        )
                except FileNotFoundError:
                    print("Error analyzing {}: temp_analysis missing.".format(out_file))
            print()
            print(
                f"===================================== END SAMPLE {samp+1} / {n_samples_per_piece} ==========================================="
            )
            print(
                "==============================================================================================="
            )

    print(
        "[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})".format(
            n_pieces * n_samples_per_piece, np.mean(times), np.std(times) # type: ignore
        )
    )
    print(
        "[entropy] {:.4f} (+/- {:.4f})".format(
            np.mean(piece_entropies), np.std(piece_entropies)
        )
    )
