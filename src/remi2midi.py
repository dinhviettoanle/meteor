import copy
import json
import os
import pickle
import random

import miditoolkit
import numpy as np

from .custom_data.constants import GEMINI_INSTR_TO_PROGRAM, MMT_MAPPING_PATH

##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16
IDX_TO_PITCH_CLASS = {
  0: 'C', 
  1: 'C#', 
  2: 'D', 
  3: 'D#', 
  4: 'E', 
  5: 'F', 
  6: 'F#', 
  7: 'G', 
  8: 'G#', 
  9: 'A', 
  10: 'A#', 
  11: 'B'
}

PITCH_CLASS_TO_IDX = {
  v:k for k, v in IDX_TO_PITCH_CLASS.items()
}


##############################
# containers for conversion
##############################
class ConversionEvent(object):
  def __init__(self, event, is_full_event=False):
    if not is_full_event:
      if 'Note_PitchClass' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Note_Octave' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Note' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Description_Chord' in event:
        self.name, self.value = '_'.join(event.split('_')[:2]), '_'.join(event.split('_')[2:])
      elif 'Description_Track' in event:
        self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      # elif 'Description_CountTracks' in event:
      #   self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      elif 'BeginTrack' in event or 'EndTracks' in event:
        self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      elif 'BeginChords' in event or 'EndChords' in event:
        self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      elif 'MelodyBegin' in event or 'MelodyEnd' in event:
        self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      elif 'Chord' in event:
        self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
      elif 'BeginPitchRange' in event or 'EndPitchRange' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Description_PitchRange' in event:
        self.name, self.value = '_'.join(event.split('_')[:2]), event.split('_')[2:]
      elif 'BeginRepeatability' in event or 'EndRepeatability' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Description_Repeatability' in event:
        self.name, self.value = '_'.join(event.split('_')[:2]), event.split('_')[2:]
      elif 'BeginMelInstr' in event or 'EndMelInstr' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Description_MelInstr' in event:
        self.name, self.value = event.rsplit('_', 1)[0], '_'.join(event.rsplit('_', 1)[1:])
      else:
        self.name, self.value = event.split('_')
    else:
      self.name, self.value = event['name'], event['value']
  def __repr__(self):
    return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
  def __init__(self, pitch, bar, position, duration, velocity, program):
    self.pitch = pitch
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
    self.duration = duration
    self.velocity = velocity
    self.program = program
  
class TempoEvent(object):
  def __init__(self, tempo, bar, position):
    self.tempo = tempo
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

class ChordEvent(object):
  def __init__(self, chord_val, bar, position):
    self.chord_val = chord_val
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

##############################
# conversion functions
##############################
def read_generated_txt(generated_path):
  f = open(generated_path, 'r')
  return f.read().splitlines()


def pitchclass_octave_to_midi(pitchclass, octave):
  return (int(octave) + 1) * 12 + PITCH_CLASS_TO_IDX[pitchclass]


def remi2midi(
    events, 
    output_midi_path=None, 
    is_full_event=False, 
    return_first_tempo=False, 
    return_chords=False, 
    enforce_tempo=False, 
    enforce_tempo_val=None, 
    multitrack_mapping=None
  ):
  if multitrack_mapping is None:
    print("[info] Using Gemini mapping")
    instr2program = GEMINI_INSTR_TO_PROGRAM
  elif multitrack_mapping == 'mmt':
    print("[info] Using MMT mapping")
    with open(MMT_MAPPING_PATH) as f:
      mmt_mapping = json.load(f)
    instr2program = mmt_mapping['instrument_program_map']
    
  program2instr = dict((p, n) for n, p in instr2program.items())  
  
  events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
  # print(events[:20])

  assert events[0].name == 'Bar'
  temp_notes = []
  temp_tempos = []
  temp_chords = []
  programs = set()

  cur_bar = 0
  cur_position = 0
  cur_track_program = 0

  for i in range(len(events)):
    if events[i].name == 'Bar':
      if i > 0:
        cur_bar += 1
    elif events[i].name == 'Beat':
      cur_position = int(events[i].value)
      assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
    elif events[i].name == 'Tempo':
      temp_tempos.append(TempoEvent(
        int(events[i].value), cur_bar, cur_position
      ))
    elif events[i].name == 'Note_Pitch' and \
         (i+1) < len(events) and events[i+1].name == 'Note_Velocity' and \
         (i+2) < len(events) and events[i+2].name == 'Note_Duration':
      # check if the 3 events are of the same instrument
      temp_notes.append(
        NoteEvent(
          pitch=int(events[i].value), 
          bar=cur_bar, position=cur_position, 
          duration=int(events[i+2].value), velocity=int(events[i+1].value),
          program=cur_track_program,
        )
      )
      programs.add(cur_track_program)
    elif events[i].name == 'Note_PitchClass' and \
         (i+1) < len(events) and events[i+1].name == 'Note_Octave' and \
         (i+2) < len(events) and events[i+2].name == 'Note_Velocity' and \
         (i+3) < len(events) and events[i+3].name == 'Note_Duration':
      # check if the 4 events are of the same instrument
      midi_note = pitchclass_octave_to_midi(events[i].value, events[i+1].value)
      temp_notes.append(
        NoteEvent(
          pitch=midi_note, 
          bar=cur_bar, position=cur_position, 
          duration=int(events[i+3].value), velocity=int(events[i+2].value),
          program=cur_track_program,
        )
      )
      programs.add(cur_track_program)
    elif events[i].name in ['Note_Velocity', 'Note_Duration', 'Note_Octave']: 
      continue
    elif events[i].name == 'Chord':
      temp_chords.append(
        ChordEvent(events[i].value, cur_bar, cur_position)
      )
    elif events[i].name == 'Track':
      cur_track_program = instr2program[events[i].value]
    elif events[i].name in ['EOS', 'PAD']:
      continue
    elif events[i].name in ['Description_BeginTracks', 'Description_EndTracks', 'Description_Track']:
      continue
    elif events[i].name in ['Description_BeginChords', 'Description_EndChords', 'Description_Chord']:
      continue
    elif events[i].name in ['Description_BeginPitchRange', 'Description_EndPitchRange', 'Description_PitchRange']:
      continue
    elif events[i].name in ['MelodyBegin', 'MelodyEnd']:
      continue
    # else:
    #   raise ValueError('Token not recognized: {}'.format(events[i]))
  
  # print (len(temp_tempos), len(temp_notes))
  midi_obj = miditoolkit.midi.parser.MidiFile() # type: ignore
  temp_instr = dict(
    (program, miditoolkit.Instrument(program=program, is_drum=False, name=program2instr[program]))
    for program in program2instr if program in programs # just to ensure an orchestral order...
  )
  
  for n in temp_notes:
    temp_instr[n.program].notes.append(
      miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )

  midi_obj.instruments = list(temp_instr.values())


  if enforce_tempo is False:
    for t in temp_tempos:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )
  else:
    if enforce_tempo_val is None:
      enforce_tempo_val = temp_tempos[1]
    for t in enforce_tempo_val:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )

  
  for c in temp_chords:
    midi_obj.markers.append(
      miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
    )
  for b in range(cur_bar):
    midi_obj.markers.append(
      miditoolkit.Marker('Bar-{}'.format(b+1), int(DEFAULT_BAR_RESOL * b))
    )

  if output_midi_path is not None:
    midi_obj.dump(output_midi_path)

  if return_first_tempo:
    return midi_obj, temp_tempos
  elif return_chords:
    return midi_obj, temp_chords
  else:
    return midi_obj
  

def copy_paste_track(output_midi_path, initial_midi_obj, generated_midi_obj, track_id):
  new_instr = initial_midi_obj.instruments[track_id]
  new_instr.program = 82 # Lead
  # new_instr.program = 6 # Harpsichord
  new_instr.name = 'melody'
  
  new_generated_midi_obj = copy.deepcopy(generated_midi_obj)
  new_generated_midi_obj.instruments = [new_instr] + generated_midi_obj.instruments
  new_generated_midi_obj.dump(output_midi_path)
  print(f"Output with melody: {output_midi_path}")