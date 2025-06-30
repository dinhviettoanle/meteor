import math
import os
import pickle
import random
import sys
from glob import glob
from typing import Dict, Iterable, List, Optional, Sized, Tuple, Union
from numpy.typing import ArrayLike


import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


IDX_TO_KEY = {
    0: "A",
    1: "A#",
    2: "B",
    3: "C",
    4: "C#",
    5: "D",
    6: "D#",
    7: "E",
    8: "F",
    9: "F#",
    10: "G",
    11: "G#",
}

KEY_TO_IDX = {v: k for k, v in IDX_TO_KEY.items()}

IDX_TO_PITCH_CLASS = {
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

PITCH_CLASS_TO_IDX = {v: k for k, v in IDX_TO_PITCH_CLASS.items()}


def pickle_load(path):
    return pickle.load(open(path, "rb"))


def get_chord_tone(chord_event):
    tone = chord_event["value"].split("_")[0]
    return tone


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def convert_event(event_seq: List, event2idx: Dict) -> List:
    if isinstance(event_seq[0], dict):
        event_seq = [
            event2idx["{}_{}".format(e["name"], e["value"])] for e in event_seq
        ]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    return event_seq


def transpose_chord(chord_event: Dict, n_keys: int) -> Dict:
    """Transpose a chord event

    Parameters
    ----------
    chord_event : Dict
        Initial chord event
    n_keys : int
        Shift in semitones

    Returns
    -------
    Dict
        Transposed chord event
    """
    if chord_event["value"] == "N_N":
        return chord_event

    orig_tone = get_chord_tone(chord_event)
    orig_tone_idx = KEY_TO_IDX[orig_tone]
    new_tone_idx = (orig_tone_idx + 12 + n_keys) % 12
    new_chord_value = chord_event["value"].replace(
        "{}_".format(orig_tone), "{}_".format(IDX_TO_KEY[new_tone_idx])
    )
    new_chord_event = {"name": chord_event["name"], "value": new_chord_value}
    # print ('keys={}. {} --> {}'.format(n_keys, chord_event, new_chord_event))

    return new_chord_event


def check_extreme_pitch(raw_events: List) -> Tuple[int, int]:
    """Get the lowest and highest pitch of a list of events
    (used for pitch augmentation, to avoid going outside range)

    Parameters
    ----------
    raw_events : List
        List of events

    Returns
    -------
    Tuple[int, int]
        lowest note, highest note
    """
    low, high = 128, 0
    for ii, ev in enumerate(raw_events):
        if ev["name"] == "Note_Pitch":
            low = min(low, int(ev["value"]))
            high = max(high, int(ev["value"]))
        elif ev["name"] == "Note_PitchClass":
            assert (
                raw_events[ii + 1]["name"] == "Note_Octave"
            ), "Next event is {}".format(raw_events[ii + 1])
            midi_value = (
                max(0, int(raw_events[ii + 1]["value"])) * 12
                + PITCH_CLASS_TO_IDX[ev["value"]]
            )
            low = min(low, midi_value)
            high = max(high, midi_value)

    return low, high


def transpose_events(raw_events: List, n_keys: int, skip_tracks: bool = False) -> List:
    """Transpose a list of events (ie only Pitch-related tokens are affected)

    Parameters
    ----------
    raw_events : List
        List of events
    n_keys : int
        Transposition shift in semitones
    skip_tracks : bool, optional
        Skip track controllability pitch range tokens, by default False

    Returns
    -------
    List
        List of transposed events
    """
    transposed_raw_events = []
    augment_octave = 0
    is_note = False
    current_track = None
    dict_description_pitch_range = dict()

    for ii, ev in enumerate(raw_events):
        if ev["name"] == "Note_Pitch":
            transposed_raw_events.append(
                {"name": ev["name"], "value": ev["value"] + n_keys}
            )
            assert not (skip_tracks) or current_track is not None
            if not (skip_tracks) and len(dict_description_pitch_range) > 0:
                dict_description_pitch_range[current_track]["previous_pitches"].append(
                    ev["value"]
                )
                dict_description_pitch_range[current_track]["new_pitches"].append(
                    ev["value"] + n_keys
                )
        elif ev["name"] == "Note_PitchClass":
            new_pitchclass = PITCH_CLASS_TO_IDX[ev["value"]] + n_keys
            transposed_raw_events.append(
                {"name": ev["name"], "value": IDX_TO_PITCH_CLASS[new_pitchclass % 12]}
            )
            augment_octave = new_pitchclass // 12
            is_note = True
        elif ev["name"] == "Note_Octave":
            assert is_note  # Check that it has just seen a pitchclass
            transposed_raw_events.append(
                {"name": ev["name"], "value": ev["value"] + augment_octave}
            )
            is_note = False
        elif ev["name"] == "Chord":
            transposed_raw_events.append(transpose_chord(ev, n_keys))
        elif ev["name"] == "Description_Chord":
            transposed_raw_events.append(transpose_chord(ev, n_keys))
        # Not pitch-related tokens
        else:
            if ev["name"] == "Bar":
                if not (skip_tracks) and len(dict_description_pitch_range) > 0:
                    for instr, dict_data in dict_description_pitch_range.items():
                        new_avg = min(130, roundup(np.mean(dict_data["new_pitches"])))
                        transposed_raw_events[dict_data["pos"]] = {
                            "name": "Description_PitchRange",
                            "value": f"{instr}-{new_avg}",
                        }
                dict_description_pitch_range = dict()
                current_track = None
            if ev["name"] == "Track":
                current_track = ev["value"]
            elif ev["name"] == "Description_PitchRange":
                instr = "-".join(ev["value"].split("-")[:-1])
                dict_description_pitch_range[instr] = {
                    "pos": ii,
                    "previous_pitches": [],
                    "new_pitches": [],
                }
            transposed_raw_events.append(ev)

    assert len(transposed_raw_events) == len(raw_events)
    return transposed_raw_events


def pitchclass_to_absolute(events: List) -> Tuple[List, Dict]:
    """Convert a list of events with pitchclass+octave to absolute pitch

    Parameters
    ----------
    events : List
        List of events in pitchclass+octave

    Returns
    -------
    Tuple[List, Dict]
        List of events in absolute pitch,
        Dictionary linking positions of pitchclass+octave tokens to new absolute tokens
    """
    absolute_pitch_events = []
    pc_to_abs_pos = dict()
    pointer_abs = 0
    is_note = True
    current_pitchclass = None
    for ii_pc, ev in enumerate(events):
        if ev["name"] == "Note_PitchClass":
            current_pitchclass = ev["value"]
            pc_to_abs_pos[ii_pc] = pointer_abs
            is_note = True
        elif ev["name"] == "Note_Octave":
            assert (
                is_note and current_pitchclass is not None
            )  # Check that it has just seen a pitchclass
            midi_value = (ev["value"] + 1) * 12 + PITCH_CLASS_TO_IDX[current_pitchclass]
            absolute_pitch_events.append({"name": "Note_Pitch", "value": midi_value})
            pc_to_abs_pos[ii_pc] = pointer_abs
            pointer_abs += 1
            is_note = False
            current_pitchclass = None
        else:
            pc_to_abs_pos[ii_pc] = pointer_abs
            pointer_abs += 1
            absolute_pitch_events.append(ev)

    return absolute_pitch_events, pc_to_abs_pos


# ========================================================================================================


class REMIFullSongTransformerDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        vocab_file: str,
        model_enc_seqlen: int = 128,
        model_dec_seqlen: int = 1280,
        model_max_bars: int = 16,
        pieces: List = [],
        do_augment: bool = True,
        augment_range: range = range(-6, 7),
        min_pitch: int = 22,
        max_pitch: int = 107,
        pad_to_same: bool = True,
        use_attr_cls: bool = True,
        use_attr_multitrack_cls: bool = True,
        appoint_st_bar: Optional[int] = None,  # Decrepated
        dec_end_pad_value: Optional[str] = None,
        melody_first: bool = False,
        melody_mask_dir: Optional[str] = None,
        return_melody_tokens: bool = False,
        asymetric_pitch_encoding: bool = False,
        files_limit: Optional[int] = None,
        gen_from_piece: Optional[str] = None,
        gen_from_bar: Optional[int] = None,
        force_return_melody: bool = False,
    ):
        """Dataloader for training w/o melody or inference

        Parameters
        ----------
        data_dir : str
            Directory with all the precomputed remi .pkl
        vocab_file : str
            Path to vocab .pkl file
        model_enc_seqlen : int, optional
            Sequence length for a bar in the encoder, by default 128
        model_dec_seqlen : int, optional
            Sequence length for the decoder, by default 1280
        model_max_bars : int, optional
            Number of bars given to the encoder (or generated by the decoder), by default 16
        pieces : List, optional
            Indexes of pieces to be in the dataset (depending on train/valid/test or inference), by default []
        do_augment : bool, optional
            Data augmentation on pitch, by default True
        augment_range : range, optional
            Semitones to augment, by default range(-6, 7)
        min_pitch : int, optional
            Minimum MIDI value, by default 22
        max_pitch : int, optional
            Maximum MIDI value, by default 107
        pad_to_same : bool, optional
            Pad sequences to same size, by default True
        use_attr_cls : bool, optional
            Use polyphonicity and rhythmicity classes, by default True
        use_attr_multitrack_cls : bool, optional
            [Decrepated] Use rhythmic and harminic diversities classes, by default True
        appoint_st_bar : Optional[int], optional
            [Decrepated], by default None
        dec_end_pad_value : Optional[str], optional
            Value to use for padding sequence for the encoder (either EOS or PAD), by default None
        melody_first : bool, optional
            Melody tokens are put in the header or not, by default False
        melody_mask_dir : Optional[str], optional
            Directory to melody mask files, by default None
        return_melody_tokens : bool, optional
            Return melodic tokens when getting item, by default False
        asymetric_pitch_encoding : bool, optional
            [Decrepated] Have the encoder and decoder handle absolute or pitchclass tokenizations, by default False
        files_limit : Optional[int], optional
            Limit of file number among the considered pieces, by default None
        gen_from_piece : Optional[str], optional
            For inference, index of the piece as the reference, by default None
        gen_from_bar : Optional[int], optional
            For inference, starting bar used as the reference, by default None
        force_return_melody : bool, optional
            Return melody tokens as a field in the dict, by default False
        """
        self.vocab_file = vocab_file
        self.melody_first = melody_first
        self.force_return_melody = force_return_melody

        self.read_vocab()

        self.data_dir = data_dir
        self.melody_mask_dir = melody_mask_dir

        if files_limit:
            self.pieces = pieces[:files_limit]
        else:
            self.pieces = pieces
        self.gen_from_piece = gen_from_piece  # gen user piece
        self.gen_from_bar = gen_from_bar  # gen user bar
        self.build_dataset()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars

        self.do_augment = do_augment
        self.augment_range = augment_range
        self.min_pitch, self.max_pitch = min_pitch, max_pitch
        self.pad_to_same = pad_to_same
        self.use_attr_cls = use_attr_cls
        self.use_attr_multitrack_cls = use_attr_multitrack_cls
        self.return_melody_tokens = return_melody_tokens
        self.asymetric_pitch_encoding = asymetric_pitch_encoding

        self.appoint_st_bar = appoint_st_bar
        if dec_end_pad_value is None:
            self.dec_end_pad_value = self.pad_token
        elif dec_end_pad_value == "EOS":
            self.dec_end_pad_value = self.eos_token
        else:
            self.dec_end_pad_value = self.pad_token

    def read_vocab(self):
        """Sets the vocabulary from the vocab file"""
        vocab = pickle_load(self.vocab_file)[0]
        self.idx2event = pickle_load(self.vocab_file)[1]
        orig_vocab_size = len(vocab)
        self.event2idx = vocab
        self.bar_token = self.event2idx["Bar_None"]
        self.eos_token = self.event2idx["EOS_None"]
        self.pad_token = orig_vocab_size
        self.vocab_size = self.pad_token + 1
        if self.melody_first:
            self.start_melody_token = self.event2idx.get("MelodySeqBegin_None")
            self.end_melody_token = self.event2idx.get("MelodySeqEnd_None")

    def build_dataset(self):
        """Gets the corresponding pieces from the precomputed files"""
        if not self.pieces:
            self.pieces = sorted(glob(os.path.join(self.data_dir, "*.pkl")))
            if self.melody_first or self.force_return_melody:
                assert (
                    self.melody_mask_dir is not None
                ), "Melody mask_dir is unavailable"
                self.melody_masks = sorted(
                    glob(os.path.join(self.melody_mask_dir, "*.pkl"))
                )
        else:
            if self.melody_first or self.force_return_melody:
                assert (
                    self.melody_mask_dir is not None
                ), "Melody mask_dir is unavailable"
                self.melody_masks = sorted(
                    [os.path.join(self.melody_mask_dir, p) for p in self.pieces]
                )
            self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])

        if self.gen_from_piece is not None:
            print("[sampled pieces] Using user piece_id:", self.gen_from_piece)
            self.pieces = [
                p
                for p in self.pieces
                if str(self.gen_from_piece) == p.split("/")[-1].replace(".pkl", "")
            ]
            assert (
                len(self.pieces) == 1
            ), "More than 1 piece found or {} not in test set -> {}".format(
                self.gen_from_piece, self.pieces
            )
            if self.melody_first or self.force_return_melody:
                self.melody_masks = [
                    p for p in self.melody_masks if self.gen_from_piece in p
                ]

        self.piece_bar_pos = []

        if self.gen_from_piece is None:
            pbar = tqdm(total=len(self.pieces), desc="Building dataset")

        for i, p in enumerate(self.pieces):
            bar_pos, p_evs = pickle_load(p)
            # if not i % 200:
            #   print ('[preparing data] now at #{}'.format(i))
            if bar_pos[-1] == len(p_evs):
                print("piece {}, got appended bar markers".format(p))
                bar_pos = bar_pos[:-1]
            if len(p_evs) - bar_pos[-1] == 2:
                # got empty trailing bar
                bar_pos = bar_pos[:-1]

            bar_pos.append(len(p_evs))

            self.piece_bar_pos.append(bar_pos)

            if self.gen_from_piece is None:
                pbar.update(1)

    def get_sample_from_file(self, piece_idx: int) -> Tuple:
        """Gets a reference from a piece starting at a random bar

        Parameters
        ----------
        piece_idx : int
            Piece index

        Returns
        -------
        Tuple
            (List of events,
            Starting bar,
            Positions of bars in the picked reference,
            Number of bars)
        """
        piece_evs = pickle_load(self.pieces[piece_idx])[1]
        if self.gen_from_bar is not None:
            picked_st_bar = self.gen_from_bar
        elif (
            len(self.piece_bar_pos[piece_idx]) > self.model_max_bars
            and self.appoint_st_bar is None
        ):
            picked_st_bar = random.choice(
                range(len(self.piece_bar_pos[piece_idx]) - self.model_max_bars)
            )
        elif (
            self.appoint_st_bar is not None
            and self.appoint_st_bar
            < len(self.piece_bar_pos[piece_idx]) - self.model_max_bars
        ):
            picked_st_bar = self.appoint_st_bar
        else:
            picked_st_bar = 0

        piece_bar_pos = self.piece_bar_pos[piece_idx]

        if len(piece_bar_pos) > self.model_max_bars:
            piece_evs = piece_evs[
                piece_bar_pos[picked_st_bar] : piece_bar_pos[
                    picked_st_bar + self.model_max_bars
                ]
            ]
            picked_bar_pos = (
                np.array(
                    piece_bar_pos[picked_st_bar : picked_st_bar + self.model_max_bars]
                )
                - piece_bar_pos[picked_st_bar]
            )
            n_bars = self.model_max_bars
        else:
            picked_bar_pos = np.array(
                piece_bar_pos
                + [piece_bar_pos[-1]] * (self.model_max_bars - len(piece_bar_pos))
            )
            n_bars = len(piece_bar_pos)
            assert len(picked_bar_pos) == self.model_max_bars

        return piece_evs, picked_st_bar, picked_bar_pos, n_bars

    def pad_sequence(
        self, seq: List, maxlen: int, pad_value: Optional[int] = None
    ) -> List:
        """Pads sequence to a fixed length

        Parameters
        ----------
        seq : List
            List of events
        maxlen : int
            Length of the output
        pad_value : Optional[int], optional
            Padding value, by default None

        Returns
        -------
        List
            List of events padded to the queried length
        """
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend([pad_value for _ in range(maxlen - len(seq))])

        return seq

    def pitch_augment(self, bar_events: List) -> Tuple[List, int]:
        """Randomly data augment on pitch tokens

        Parameters
        ----------
        bar_events : List
            List of events

        Returns
        -------
        Tuple[List, int]
            (List of events transposed,
            Number of semitones)
        """
        bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)

        n_keys = random.choice(self.augment_range)
        n_tries = 0
        while (
            bar_min_pitch + n_keys < self.min_pitch
            or bar_max_pitch + n_keys > self.max_pitch
        ):
            n_keys = random.choice(self.augment_range)
            n_tries += 1
            if n_tries > 20:
                n_keys = 0
                break

        augmented_bar_events = transpose_events(bar_events, n_keys)
        return augmented_bar_events, n_keys

    def get_attr_classes(self, piece: str, st_bar: int) -> Tuple[List, List]:
        """Get polyphonicity and rhythmicity classes

        Parameters
        ----------
        piece : str
            Name of the pkl file
        st_bar : int
            Starting bar

        Returns
        -------
        Tuple[List, List]
            (Polyphonicity classes,
            Rhythmicity classes)
        """
        polyph_cls = pickle_load(os.path.join(self.data_dir, "attr_cls/polyph", piece))[
            st_bar : st_bar + self.model_max_bars
        ]
        rfreq_cls = pickle_load(os.path.join(self.data_dir, "attr_cls/rhythm", piece))[
            st_bar : st_bar + self.model_max_bars
        ]

        polyph_cls.extend([0 for _ in range(self.model_max_bars - len(polyph_cls))])
        rfreq_cls.extend([0 for _ in range(self.model_max_bars - len(rfreq_cls))])

        assert len(polyph_cls) == self.model_max_bars
        assert len(rfreq_cls) == self.model_max_bars

        return polyph_cls, rfreq_cls

    def get_attr_polyph_classes(self, piece: str, st_bar: int) -> Tuple[List, List]:
        """[Decrepated] Get rhythmic and harmonic diversity classes

        Parameters
        ----------
        piece : str
            Name of the pkl file
        st_bar : int
            Starting bar

        Returns
        -------
        Tuple[List, List]
            (Harmonic diversity classes,
            Rhythmic diversity classes)
        """
        harmdiv_cls = pickle_load(
            os.path.join(self.data_dir, "attr_cls/harmdiv", piece)
        )[st_bar : st_bar + self.model_max_bars]
        rhythmdiv_cls = pickle_load(
            os.path.join(self.data_dir, "attr_cls/rhythmdiv", piece)
        )[st_bar : st_bar + self.model_max_bars]

        harmdiv_cls.extend([0 for _ in range(self.model_max_bars - len(harmdiv_cls))])
        rhythmdiv_cls.extend(
            [0 for _ in range(self.model_max_bars - len(rhythmdiv_cls))]
        )

        assert len(harmdiv_cls) == self.model_max_bars
        assert len(rhythmdiv_cls) == self.model_max_bars

        return harmdiv_cls, rhythmdiv_cls

    def get_encoder_input_data(self, bar_positions: List, bar_events: List) -> Tuple:
        """Get the music data (no melody, classes) and format it for the encoder

        Parameters
        ----------
        bar_positions : List
            Indexes of the bars
        bar_events : List
            List of flattened events

        Returns
        -------
        Tuple
            (Encoder input,
            Padding indexes as a mask,
            Lenghts of each token sequence per bar)
        """
        assert len(bar_positions) == self.model_max_bars + 1
        enc_padding_mask = np.ones(
            (self.model_max_bars, self.model_enc_seqlen), dtype=bool
        )
        enc_padding_mask[:, :2] = False
        padded_enc_input = np.full(
            (self.model_max_bars, self.model_enc_seqlen),
            dtype=int,
            fill_value=self.pad_token,
        )
        enc_lens = np.zeros((self.model_max_bars,))

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            enc_padding_mask[b, : (ed - st)] = False
            enc_lens[b] = ed - st
            within_bar_events = self.pad_sequence(
                bar_events[st:ed], self.model_enc_seqlen, self.pad_token
            )
            within_bar_events = np.array(within_bar_events)

            padded_enc_input[b, :] = within_bar_events[: self.model_enc_seqlen]

        return padded_enc_input, enc_padding_mask, enc_lens

    def get_melody_events_encoder_input_data(
        self,
        bar_positions: List,
        bar_events: List,
        melody_bar_positions: List,
        melody_events: List,
    ) -> Tuple:  # Decrepated
        """[Decrepated] Get the music and melody data

        Parameters
        ----------
        bar_positions : List
            Positions of bars for the music
        bar_events : List
            Events per bar
        melody_bar_positions : List
            Bar positions for the melody
        melody_events : List
            Melodic events

        Returns
        -------
        Tuple
            (Encoder input,
            Padding indexes as a mask,
            Lenghts of each token sequence per bar,
            Melodic tokens)
        """
        assert len(bar_positions) == self.model_max_bars + 1
        assert len(melody_bar_positions) == self.model_max_bars + 1

        enc_padding_mask = np.ones(
            (self.model_max_bars, self.model_enc_seqlen), dtype=bool
        )
        enc_padding_mask[:, :2] = False
        padded_enc_input = np.full(
            (self.model_max_bars, self.model_enc_seqlen),
            dtype=int,
            fill_value=self.pad_token,
        )
        melody_tokens_only = []
        enc_lens = np.zeros((self.model_max_bars,))

        for b, (st, ed, m_st, m_ed) in enumerate(
            zip(
                bar_positions[:-1],
                bar_positions[1:],
                melody_bar_positions[:-1],
                melody_bar_positions[1:],
            )
        ):
            seq_len = (ed - st) + (m_ed - m_st)
            enc_padding_mask[b, :seq_len] = False
            enc_lens[b] = seq_len
            mel_seq = (
                [self.start_melody_token]
                + melody_events[m_st:m_ed]
                + [self.end_melody_token]
            )
            melody_tokens_only.append(
                melody_events[m_st + 1 : m_ed]
            )  # +1 to skip bar event
            within_bar_events = self.pad_sequence(
                mel_seq + bar_events[st:ed], self.model_enc_seqlen, self.pad_token
            )
            within_bar_events = np.array(within_bar_events)

            padded_enc_input[b, :] = within_bar_events[: self.model_enc_seqlen]

        return padded_enc_input, enc_padding_mask, enc_lens, melody_tokens_only

    def get_melody_events(self, piece_idx: int, start_bar: int, n_bars: int) -> Tuple:
        """[Decrepated] Get the melody events only (no music, controls) from a file (slower)

        Parameters
        ----------
        piece_idx : int
            Index of the piece
        start_bar : int
            Starting bar
        n_bars : int
            Number of bars requested

        Returns
        -------
        Tuple
            (List of melodic events,
            Bar positions within this list of melodic events)
        """
        melody_mask_data = pickle_load(
            self.melody_masks[piece_idx]
        )  # keys: melody_bar_pos, mask, melody_tokens
        melody_bar_pos = melody_mask_data["melody_bar_pos"]
        assert len(self.piece_bar_pos[piece_idx]) == len(
            melody_bar_pos
        ), "In piece: {} ; In melody: {}".format(
            len(self.piece_bar_pos[piece_idx]), len(melody_bar_pos)
        )

        melody_events = melody_mask_data["melody_events"]

        if len(melody_bar_pos) > self.model_max_bars:
            melody_events = melody_mask_data["melody_events"][
                melody_bar_pos[start_bar] : melody_bar_pos[start_bar + n_bars]
            ]
            picked_melody_bar_pos = (
                np.array(melody_bar_pos[start_bar : start_bar + n_bars])
                - melody_bar_pos[start_bar]
            )
        else:
            picked_melody_bar_pos = np.array(
                melody_bar_pos
                + [melody_bar_pos[-1]] * (self.model_max_bars - len(melody_bar_pos))
            )
            assert len(picked_melody_bar_pos) == self.model_max_bars

        return melody_events, picked_melody_bar_pos

    def get_melody_events_only(
        self, bar_positions: List, melody_bar_positions: List, melody_events: List
    ) -> List:
        """[Decrepated] Get the melody events only (no music, controls) from the list of events (faster)
        (ie take everything except bar events...)

        Parameters
        ----------
        bar_positions : List
            Indexes of bar tokens in the music
        melody_bar_positions : List
            Indexes of bar in the melody tokens
        melody_events : List
            List of melody events

        Returns
        -------
        List
            List of melodic events
        """
        assert len(bar_positions) == self.model_max_bars + 1
        assert len(melody_bar_positions) == self.model_max_bars + 1

        melody_tokens_only = []
        for b, (st, ed, m_st, m_ed) in enumerate(
            zip(
                bar_positions[:-1],
                bar_positions[1:],
                melody_bar_positions[:-1],
                melody_bar_positions[1:],
            )
        ):
            melody_tokens_only.append(
                melody_events[m_st + 1 : m_ed]
            )  # +1 to skip bar event

        return melody_tokens_only

    def make_bar_beat_melody_dict(self, melody_tokens: List) -> Dict:
        """Makes the dictionary of melodic tokens used for inference guidance

        Parameters
        ----------
        melody_tokens : List
            List of melodic tokens

        Returns
        -------
        Dict
            Dictionary indexes as {
                'bar_id': {
                    'beat_id': [melodic tokens]
                }
            }
        """
        bar_beat_melody = dict()
        for bar_idx, bar_melody_tokens in enumerate(melody_tokens):
            bar_beat_melody[bar_idx] = dict()
            current_beat = 0
            for beat_idx, beat_melody_token in enumerate(bar_melody_tokens):
                str_token = self.idx2event[beat_melody_token]
                if "Beat" in str_token:
                    current_beat = int(str_token.split("_")[-1])
                    bar_beat_melody[bar_idx][current_beat] = []
                else:
                    if str_token not in bar_beat_melody[bar_idx][current_beat]:
                        # bar_beat_melody[bar_idx][current_beat].append(beat_melody_token)
                        bar_beat_melody[bar_idx][current_beat].append(str_token)

        return bar_beat_melody

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bar_events, st_bar, bar_pos, enc_n_bars = self.get_sample_from_file(idx)

        melody_events = []
        if self.melody_first or self.force_return_melody:
            melody_events, melody_bar_pos = self.get_melody_events(
                idx, st_bar, enc_n_bars
            )

        if self.do_augment:
            bar_events, n_keys = self.pitch_augment(bar_events)
            if self.melody_first or self.force_return_melody:
                melody_events = transpose_events(melody_events, n_keys)

        # control attributes classes
        if self.use_attr_cls:
            polyph_cls, rfreq_cls = self.get_attr_classes(
                os.path.basename(self.pieces[idx]), st_bar
            )
            polyph_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            rfreq_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                polyph_cls_expanded[b_st:b_ed] = polyph_cls[i]
                rfreq_cls_expanded[b_st:b_ed] = rfreq_cls[i]
        else:
            polyph_cls, rfreq_cls = [0], [0]
            polyph_cls_expanded, rfreq_cls_expanded = [0], [0]

        # control polyphonic attributes classes
        if self.use_attr_multitrack_cls:
            harmdiv_cls, rhythmdiv_cls = self.get_attr_polyph_classes(
                os.path.basename(self.pieces[idx]), st_bar
            )
            harmdiv_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            rhythmdiv_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                harmdiv_cls_expanded[b_st:b_ed] = harmdiv_cls[i]
                rhythmdiv_cls_expanded[b_st:b_ed] = rhythmdiv_cls[i]
        else:
            harmdiv_cls, rhythmdiv_cls = [0], [0]
            harmdiv_cls_expanded, rhythmdiv_cls_expanded = [0], [0]

        bar_tokens = convert_event(bar_events, self.event2idx)
        bar_pos = bar_pos.tolist() + [len(bar_tokens)]

        # if self.asymetric_pitch_encoding:
        # absolute_events, pc_to_abs_pos = pitchclass_to_absolute(bar_events)
        # absolute_tokens = convert_event(absolute_events, self.event2idx)

        melody_tokens_dict = dict()
        melody_tokens = list()
        if not self.melody_first:
            enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(
                bar_pos, bar_tokens
            )
            if self.force_return_melody:
                melody_tokens = convert_event(melody_events, self.event2idx)
                melody_bar_pos = melody_bar_pos.tolist() + [len(melody_tokens)]
                melody_tokens_list = self.get_melody_events_only(
                    bar_pos, melody_bar_pos, melody_tokens
                )
                melody_tokens_dict = self.make_bar_beat_melody_dict(melody_tokens_list)
        else:
            melody_tokens = convert_event(melody_events, self.event2idx)
            melody_bar_pos = melody_bar_pos.tolist() + [len(melody_tokens)]
            enc_inp, enc_padding_mask, enc_lens, melody_tokens_list = (
                self.get_melody_events_encoder_input_data(
                    bar_pos, bar_tokens, melody_bar_pos, melody_tokens
                )
            )
            if self.return_melody_tokens:
                melody_tokens_dict = self.make_bar_beat_melody_dict(melody_tokens_list)

        length = len(bar_tokens)
        if self.pad_to_same:
            inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1)
        else:
            inp = self.pad_sequence(
                bar_tokens, len(bar_tokens) + 1, pad_value=self.dec_end_pad_value
            )
        target = np.array(inp[1:], dtype=int)
        inp = np.array(inp[:-1], dtype=int)
        assert len(inp) == len(target)

        return {
            "id": idx,
            "piece_id": int(os.path.basename(self.pieces[idx]).replace(".pkl", "")),
            "st_bar_id": st_bar,
            "bar_pos": np.array(bar_pos, dtype=int),
            "enc_input": enc_inp,
            "dec_input": inp[: self.model_dec_seqlen],
            "dec_target": target[: self.model_dec_seqlen],
            # control classes
            "polyph_cls": polyph_cls_expanded,
            "rhymfreq_cls": rfreq_cls_expanded,
            "polyph_cls_bar": np.array(polyph_cls),
            "rhymfreq_cls_bar": np.array(rfreq_cls),
            # polyph
            "harmdiv_cls": harmdiv_cls_expanded,
            "rhythmdiv_cls": rhythmdiv_cls_expanded,
            "harmdiv_cls_bar": np.array(harmdiv_cls),
            "rhythmdiv_cls_bar": np.array(rhythmdiv_cls),
            # melody
            "melody_tokens_only": melody_tokens_dict,
            "melody_tokens_list": melody_tokens,
            "melody_events": melody_events,
            # lengths
            "length": min(length, self.model_dec_seqlen),
            "enc_padding_mask": enc_padding_mask,
            "enc_length": enc_lens,
            "enc_n_bars": enc_n_bars,
        }


# %%
# ==========================================================================================================================
# ===================================================== MELODY =============================================================
# ==========================================================================================================================


def make_melody_mask_single(
    tokens: ArrayLike, melody_begin_idx: List, melody_end_idx: List, bar_beat_idxs: ArrayLike
) -> Iterable:
    """Makes the melody mask for a loader item

    Parameters
    ----------
    tokens : ArrayLike
        List of tokens
    melody_begin_idx : List
        Indexes of tokens starting melody notes
    melody_end_idx : List
        Indexes of tokens ending melody notes
    bar_beat_idxs : ArrayLike
        Indexes of bar tokens

    Returns
    -------
    Iterable
        Binary mask of tokens being the melody or not
    """
    tokens = np.array(tokens)
    melodic_mask = np.zeros(len(tokens))

    mask = np.isin(tokens, bar_beat_idxs)
    melodic_mask[mask] = 1

    melody_starts = np.where(tokens == melody_begin_idx)[0]
    melody_ends = np.where(tokens == melody_end_idx)[0]

    for start, end in zip(melody_starts, melody_ends):
        melodic_mask[start : end + 1] = 1

    return melodic_mask


class FullSongToMelodyDataset(REMIFullSongTransformerDataset):  # Only used for training
    def __init__(self, *args, **kwargs):
        """Same as REMIFullSongTransformerDataset, but returns also melody tokens
        to compute loss to reconstruct the melody
        """
        super().__init__(*args, **kwargs)

        self.melody_begin_idx = self.event2idx["MelodyBegin_None"]
        self.melody_end_idx = self.event2idx["MelodyEnd_None"]
        beat_idxs = [idx for ev, idx in self.event2idx.items() if "Beat" in ev]
        bar_idx = self.event2idx["Bar_None"]

        self.bar_beat_idxs = torch.tensor([bar_idx] + beat_idxs)

    def extract_melody(self, bar_tokens: ArrayLike) -> Tuple[Iterable, Iterable]:
        """Extract melodic tokens only and mask

        Parameters
        ----------
        bar_tokens : ArrayLike
            Tokens of a bar

        Returns
        -------
        Tuple[Iterable, Iterable]
            (Binary mask, list of melodic tokens)
        """
        melody_mask = make_melody_mask_single(
            bar_tokens, self.melody_begin_idx, self.melody_end_idx, self.bar_beat_idxs
        )
        melody_tokens = np.array(bar_tokens)[melody_mask == 1]
        return melody_mask, melody_tokens

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bar_events, st_bar, bar_pos, enc_n_bars = self.get_sample_from_file(idx)
        if self.do_augment:
            bar_events, n_keys = self.pitch_augment(bar_events)

        # control attributes classes
        if self.use_attr_cls:

            polyph_cls, rfreq_cls = self.get_attr_classes(
                os.path.basename(self.pieces[idx]), st_bar
            )
            polyph_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            rfreq_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                polyph_cls_expanded[b_st:b_ed] = polyph_cls[i]
                rfreq_cls_expanded[b_st:b_ed] = rfreq_cls[i]
        else:
            polyph_cls, rfreq_cls = [0], [0]
            polyph_cls_expanded, rfreq_cls_expanded = [0], [0]

        # control polyphonic attributes classes
        if self.use_attr_multitrack_cls:
            harmdiv_cls, rhythmdiv_cls = self.get_attr_polyph_classes(
                os.path.basename(self.pieces[idx]), st_bar
            )
            harmdiv_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            rhythmdiv_cls_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
            for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
                harmdiv_cls_expanded[b_st:b_ed] = harmdiv_cls[i]
                rhythmdiv_cls_expanded[b_st:b_ed] = rhythmdiv_cls[i]
        else:
            harmdiv_cls, rhythmdiv_cls = [0], [0]
            harmdiv_cls_expanded, rhythmdiv_cls_expanded = [0], [0]

        bar_tokens = convert_event(bar_events, self.event2idx)
        bar_pos = bar_pos.tolist() + [len(bar_tokens)]

        melody_mask, melody_tokens = self.extract_melody(bar_tokens)

        enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(
            bar_pos, bar_tokens
        )

        length = len(bar_tokens)
        if self.pad_to_same:
            inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1)
            melody_mask = self.pad_sequence(
                list(melody_mask), self.model_dec_seqlen + 1, pad_value=0
            )
            melody_tokens = self.pad_sequence(
                list(melody_tokens),
                self.model_dec_seqlen + 1,
                pad_value=self.dec_end_pad_value,
            )
        else:
            inp = self.pad_sequence(
                bar_tokens, len(bar_tokens) + 1, pad_value=self.dec_end_pad_value
            )
            melody_mask = self.pad_sequence(
                list(melody_mask), len(bar_tokens) + 1, pad_value=0
            )
            melody_tokens = self.pad_sequence(
                list(melody_tokens),
                len(bar_tokens) + 1,
                pad_value=self.dec_end_pad_value,
            )

        target = np.array(inp[1:], dtype=int)
        inp = np.array(inp[:-1], dtype=int)
        melody_mask = np.array(melody_mask[:-1], dtype=int)
        melody_tokens = np.array(melody_tokens[:-1])
        assert len(inp) == len(target)

        return {
            "id": idx,
            "piece_id": int(os.path.basename(self.pieces[idx]).replace(".pkl", "")),
            "st_bar_id": st_bar,
            "bar_pos": np.array(bar_pos, dtype=int),
            "enc_input": enc_inp,
            "dec_input": inp[: self.model_dec_seqlen],
            "dec_target": target[: self.model_dec_seqlen],
            # control classes
            "polyph_cls": polyph_cls_expanded,
            "rhymfreq_cls": rfreq_cls_expanded,
            "polyph_cls_bar": np.array(polyph_cls),
            "rhymfreq_cls_bar": np.array(rfreq_cls),
            # polyph
            "harmdiv_cls": harmdiv_cls_expanded,
            "rhythmdiv_cls": rhythmdiv_cls_expanded,
            "harmdiv_cls_bar": np.array(harmdiv_cls),
            "rhythmdiv_cls_bar": np.array(rhythmdiv_cls),
            # melody:
            "melody_mask": melody_mask[: self.model_dec_seqlen],
            "melody_tokens": melody_tokens[: self.model_dec_seqlen],
            # lengths
            "length": min(length, self.model_dec_seqlen),
            "enc_padding_mask": enc_padding_mask,
            "enc_length": enc_lens,
            "enc_n_bars": enc_n_bars,
        }
