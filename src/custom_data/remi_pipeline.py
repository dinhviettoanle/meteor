import argparse
import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional

import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .. import attributes
from . import analyzer, corpus2events, events2musemorphose, midi2corpus
from .constants import MMT_MAPPING_PATH

with open(MMT_MAPPING_PATH, encoding="utf-8") as f:
    mmt_mapping = json.load(f)
instr2program = mmt_mapping["instrument_program_map"]
instr_list = list(instr2program.keys())


seed = 1


def pickle_load(f):
    return pickle.load(open(f, "rb"))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def create_remi_vocabulary(
    event_path: str,
    vocab_path: str,
    pitch_encoding: str,
    header_pitch_range: bool,
    header_repeatability: bool,
    melodic_tokens_position: str,
    vocab_absolute_pitch: Optional[bool],
):
    """Creates the remi vocabulary file (a pickle file)

    Parameters
    ----------
    event_path : str
        Path to event files
    vocab_path : str
        Output path for the vocabulary file
    pitch_encoding : str
        Pitch encoding (absolute or pitchclass)
    header_pitch_range : bool
        Add pitch_range tokens in the header
    header_repeatability : bool
        Add reapeatability tokens in the header
    melodic_tokens_position : str
       Where is the information of melodic events situated (before (ie. in the header) or note)
    vocab_absolute_pitch : Optional[bool]
        Include absolute pitch tokens in the vocabulary or not 
        even if we consider pitchclasses

    Raises
    ------
    AttributeError
        _description_
    """
    event_files = list(Path(event_path).glob("**/*.pkl"))

    vocab = set()
    all_tracks = set()
    for f in tqdm(event_files, leave=False):
        events = pickle_load(f)
        for e in events:
            if e["name"] == "Track":
                all_tracks.add(e["value"])
            str_event = f"{e['name']}_{e['value']}"
            vocab.add(str_event)

    # To make sure that all transposition can be performed
    if pitch_encoding == "absolute" or vocab_absolute_pitch:
        for p in range(1, 128):
            str_pitch = f"Note_Pitch_{p}"
            vocab.add(str_pitch)
    elif pitch_encoding == "pitchclass":
        for pc in corpus2events.NOTES_NAMES:
            vocab.add(f"Note_PitchClass_{pc}")
        for oc in range(0, 13):
            vocab.add(f"Note_Octave_{oc}")
    else:
        raise AttributeError("Wrong pitch encoding provided: {}".format(pitch_encoding))

    if header_pitch_range:
        for track_name in all_tracks:
            for rg in range(0, 14):
                vocab.add(f"Description_PitchRange_{track_name}-{10*rg}")

    if melodic_tokens_position:
        vocab.add("MelodySeqBegin_None")
        vocab.add("MelodySeqEnd_None")

    if header_repeatability:
        for track_name in all_tracks:
            for r in range(1, 13):
                vocab.add(f"Description_Repeatability_{track_name}-{r}")

    idx2ev = dict((ii, e) for ii, e in enumerate(sorted(list(vocab))))
    ev2idx = dict((e, ii) for ii, e in idx2ev.items())

    pickle_dump((ev2idx, idx2ev), vocab_path)
    print("Vocab size:", len(ev2idx))
    print("Vocab saved at", vocab_path)


def split_sets(
    event_path: str,
    pickle_path: str,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
):
    """Makes a files that describe train / valid / test sets

    Parameters
    ----------
    event_path : str
        Path to event files
    pickle_path : str
        Output path to the final pickle file
    train_size : float, optional
        Size of train set, by default 0.8
    valid_size : float, optional
        Size of valid set, by default 0.1
    test_size : float, optional
        Size of test set, by default 0.1
    """
    Path(pickle_path).mkdir(parents=True, exist_ok=True)
    assert train_size + valid_size + test_size == 1
    files = list(Path(event_path).glob("**/*.pkl"))
    files = [f.name for f in files]
    train_files, test_files = train_test_split(
        files, test_size=test_size, random_state=seed
    )
    train_files, valid_files = train_test_split(
        train_files,
        test_size=1 - train_size / (train_size + valid_size),
        random_state=seed,
    )
    print(
        "Training: {} ; Valid: {}; Test: {}".format(
            len(train_files), len(valid_files), len(test_files)
        )
    )
    pickle_dump(train_files, pickle_path.format("_train_pieces"))
    pickle_dump(valid_files, pickle_path.format("_val_pieces"))
    pickle_dump(test_files, pickle_path.format("_test_pieces"))


# =============================================================================================


def mid2musemorphose(config: Dict):
    """Main function to call to compute features from MIDI files

    Parameters
    ----------
    config : Dict
        Configuration dictionary
    """
    print(">>> Running midi2corpus")
    # midi2corpus.main(
    midi2corpus.main_parallel(
        path_indir=config["midi_analyzed_path"],
        path_outdir=config["corpus_path"],
        multi_track=config["multi_track"],
        multi_track_mapping=config["multi_track_mapping"],
        files_limit=config.get("files_limit"),
    )

    print(">>> Running corpus2events")
    # corpus2events.main(
    corpus2events.main_parallel(
        path_indir=config["corpus_path"],
        path_outdir=config["events_path"],
        path_maskdir=config.get("melody_mask_path"),
        tokenization=config["tokenization"],
        multi_track=config["multi_track"],
        pitch_encoding=config["pitch_encoding"],
        header_pitch_range=config["header_pitch_range"],
        header_repeatability=config["header_repeatability"],
        header_chords=config["header_chords"],
        header_melodic_instr=config["header_melodic_instr"],
        melodic_tokens=config["melodic_tokens"],
        melodic_tokens_position=config["melodic_tokens_position"],
        order_first=config["order_first"],
    )

    print(">>> Running events2musemorphose")
    events2musemorphose.main(
        events_path=config["events_path"],
        musemorphose_path=config["musemorphose_path"],
        tokenization=config["tokenization"],
    )

    print(">>> Creating vocab")
    create_remi_vocabulary(
        event_path=config["events_path"],
        vocab_path=config["vocab_path"],
        pitch_encoding=config["pitch_encoding"],
        header_pitch_range=config["header_pitch_range"],
        header_repeatability=config["header_repeatability"],
        melodic_tokens_position=config["melodic_tokens_position"],
        vocab_absolute_pitch=config.get("vocab_absolute_pitch"),
    )

    print(">>> Spliting train/valid/test")
    if not (config["skip_splits"]):
        try:
            split_sets(
                event_path=config["events_path"], pickle_path=config["pickle_path"]
            )
        except ValueError:
            print("     [WARNING] Not enough data... Splits not created")
    else:
        print("     [INFO] Skipping spliting")

    compute_attributes(config)

    print("Done!")


def compute_attributes(config: Dict):
    """Compute the attributes from an already computed musemorphose features file

    Parameters
    ----------
    config : Dict
        Configuration file
    """
    print(">> Running attributes")
    if config["tokenization"] == "remi":
        attributes.main(
            data_dir=config["cls_path"],
            polyph_out_dir=os.path.join(config["cls_path"], "attr_cls/polyph"),
            rhythm_out_dir=os.path.join(config["cls_path"], "attr_cls/rhythm"),
            harmdiv_out_dir=os.path.join(config["cls_path"], "attr_cls/harmdiv"),
            rhythmdiv_out_dir=os.path.join(config["cls_path"], "attr_cls/rhythmdiv"),
            rhythm_local_out_dir=os.path.join(
                config["cls_path"], "attr_cls/local_rhythm"
            ),
            pitch_local_out_dir=os.path.join(
                config["cls_path"], "attr_cls/local_pitch"
            ),
            config_out_path=os.path.join(config["cls_path"], "attr_cls"),
            instr_list=instr_list,
        )


def analyze(config):
    print(">>> Running analyzer")
    analyzer.main(
        path_indir=config["midi_raw_path"], path_outdir=config["midi_analyzed_path"]
    )

# =============================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", required=False, action="store_true")
    parser.add_argument("--mid2musemorphose", required=False, action="store_true")
    parser.add_argument("--attributes", required=False, action="store_true")
    parser.add_argument("--all", required=False, action="store_true")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    pprint(config)

    if args.analyzer:
        analyze(config)
    elif args.mid2musemorphose:
        mid2musemorphose(config)
    elif args.attributes:
        compute_attributes(config)
    elif args.all:
        analyze(config)
        mid2musemorphose(config)
