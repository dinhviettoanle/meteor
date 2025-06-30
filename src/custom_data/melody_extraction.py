from typing import List, Tuple
import numpy as np

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


def skyline_selection(notes_in_beat: List[Tuple]) -> Tuple:
    """Select the skyline notes among a list of simultaneous notes

    Parameters
    ----------
    notes_in_beat : List[Tuple]
        List of tuples containaing
        (Pitch, Duration, Velocity, index, next_note_index)

    Returns
    -------
    Tuple
        Tuple containing the highest note within the simultaneous notes
        as ((Pitch, Duration, Velocity, index), note_start, note_end)
    """
    sorted_notes_in_beat = sorted(
        notes_in_beat, key=lambda x: x[0]["value"], reverse=True
    )
    return (
        list(sorted_notes_in_beat[0])[:-1],
        sorted_notes_in_beat[0][-2],
        sorted_notes_in_beat[0][-1],
    )


def extract_melody_events(events: List) -> Tuple:
    """Extract the melody events as beat-wise skyline

    Parameters
    ----------
    events : List
        List of events

    Returns
    -------
    Tuple
        (
            List of melody events,
            List of start indexes for melodic notes,
            List of end indexes for melodic notes
        )
    """
    melody_events = []
    indexes_melody_start = []
    indexes_melody_end = []
    notes_in_beat = []

    for ii, ev in enumerate(events):
        if ev["name"] == "Bar":
            melody_events.append(ev)
        elif ev["name"] == "Note_Pitch":
            notes_in_beat.append((ev, events[ii + 1], events[ii + 2], ii, ii + 3))
        elif ev["name"] == "Beat":
            melody_events.append(ev)
            if len(notes_in_beat) > 0:
                melodic_note, note_start, note_end = skyline_selection(notes_in_beat)
                melody_events.extend(melodic_note)
                indexes_melody_start.append(note_start)
                indexes_melody_end.append(note_end)
            notes_in_beat = []

    return melody_events, indexes_melody_start, indexes_melody_end


# ================================================================================


def extract_melody_track(
    events: List, order_first: str, KEEP_TRACK: bool = False
) -> Tuple:
    """Get the melodic tracks from a bar as bar-wise track-wise skyline 
    (ie the melodic track is the one having the higher average pitch)

    Parameters
    ----------
    events : List
        List of events
    order_first : str
        Beat-first or track-first parsing   
    KEEP_TRACK : bool, optional
        Keep <track> tokens in the melodic tokens , by default False

    Returns
    -------
    Tuple
        (
            List of start indexes for melodic notes,
            List of end indexes for melodic notes
            List of melody events,
        )
    """
    instruments_pitch_range = dict()
    track_limits = dict()
    current_track = None
    current_pitchclass = None
    indexes_melody_start = []
    indexes_melody_end = []
    all_melodic_tracks = []
    limits_diff = 0 if KEEP_TRACK else 1
    n_bars = 0

    for ii, ev in enumerate(events):
        if ev["name"] == "Bar":
            n_bars += 1
            if current_track is not None and order_first == "track":
                track_limits[current_track].append(ii)
            if len(instruments_pitch_range) > 0:
                # Compute pitch range
                instrument_mean_pitch_range = [
                    (instr, np.mean(pitches))
                    for instr, pitches in instruments_pitch_range.items()
                ]
                instrument_mean_pitch_range = sorted(
                    instrument_mean_pitch_range, key=lambda x: x[1], reverse=True
                )
                melodic_tracks = [
                    x[0]
                    for x in instrument_mean_pitch_range
                    if x[1] == instrument_mean_pitch_range[0][1]
                ]
                all_melodic_tracks.append(melodic_tracks)
                if order_first == "track":
                    indexes_melody_start.extend(
                        [track_limits[instr][0] for instr in melodic_tracks]
                    )
                    indexes_melody_end.extend(
                        [track_limits[instr][1] for instr in melodic_tracks]
                    )
                elif order_first == "beat":
                    for instr in melodic_tracks:
                        indexes_melody_start.extend(
                            [t for t in track_limits[instr][::2]]
                        )
                        indexes_melody_end.extend(
                            [t for t in track_limits[instr][1::2]]
                        )
                # Re-init variables
                instruments_pitch_range = dict()
                track_limits = dict()
                current_track = None
            elif ii > 0 and len(instruments_pitch_range) == 0:
                all_melodic_tracks.append([])
        elif ev["name"] == "Note_Pitch":
            instruments_pitch_range[current_track].append(ev["value"])
        elif ev["name"] == "Note_PitchClass":
            current_pitchclass = ev["value"]
        elif ev["name"] == "Note_Octave":
            assert (
                current_pitchclass is not None
            )  # Check that it has just seen a pitchclass
            midi_value = (ev["value"] + 1) * 12 + PITCH_CLASS_TO_IDX[current_pitchclass]
            instruments_pitch_range[current_track].append(midi_value)
            current_pitchclass = None

        # Case track -> multiple beats
        if order_first == "track":
            if ev["name"] == "Track":
                instruments_pitch_range[ev["value"]] = []
                track_limits[ev["value"]] = [ii]
                if current_track is not None:
                    track_limits[current_track].append(ii)
                current_track = ev["value"]

        # Case beat -> multiple tracks
        elif order_first == "beat":
            if ev["name"] == "Track":
                if ev["value"] not in instruments_pitch_range:
                    instruments_pitch_range[ev["value"]] = []
                if ev["value"] not in track_limits:
                    track_limits[ev["value"]] = [
                        ii + limits_diff,
                        ii + 4,
                    ]  # ii, ii+4 to include track
                else:
                    track_limits[ev["value"]].extend(
                        [ii + limits_diff, ii + 4]
                    )  # ii, ii+4 to include track
                current_track = ev["value"]

    return indexes_melody_start, indexes_melody_end, all_melodic_tracks



def make_melodic_sequence(events: List, include_chords: bool = False) -> List:
    """Only extract the melodic events encapsulated between MelodyBegin and MelodyEnd tokens

    Parameters
    ----------
    events : List
        List of events
    include_chords : bool, optional
        Include chord tokens in melody, by default False

    Returns
    -------
    List
        List of melodic events
    """
    
    melody_events = []
    is_in_melody = False

    for ii, ev in enumerate(events):
        # if ev['name'] == 'Track':
        #     continue
        if is_in_melody and ev["name"] not in ["MelodyBegin", "MelodyEnd"]:
            melody_events.append(ev)

        if ev["name"] == "Bar" or ev["name"] == "Beat":
            melody_events.append(ev)
        elif include_chords and ev["name"] == "Chord":  # Accept chords in melody
            melody_events.append(ev)
        elif ev["name"] == "MelodyBegin":
            is_in_melody = True
        elif ev["name"] == "MelodyEnd":
            is_in_melody = False

    return melody_events
