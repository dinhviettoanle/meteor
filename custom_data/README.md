# How to use custom dataset (MIDI format)

- Step 1: Put the MIDI files in `custom_data/midi_analyzed/` folder
- Step 2: Go to `custom_data/` folder and run the following scripts in order:
  - `midi2corpus.py`
  - `corpus2events.py`
  - `events2musemorphose.py`
  
```
cd custom_data
python3 midi2corpus.py
python3 corpus2events.py
python3 events2musemorphose.py
```
  
- Step 3:
  - Go to back the parent folder and run `attributes.py` 
```
cd ..
python3 attributes.py
```

- Done! Now you can generate new music by running the following comand:
```
python3 generate.py config/default.yaml musemorphose_pretrained_weights.pt generations/ [num pieces] [num samples per piece]
```


# How to use custom dataset (Mp3 format)

- Step 1: Put the mp3 files in `custom_data/mp3/` folder
- Step 2: Transcribe the mp3 files to MIDI using an external tool
- Step 3: Add the transcribed MIDI files to `custom_data/midi_transcribed/`
- Step 4: Go to `custom_data/` folder and run the following scripts in order:
  - `synchronizer.py`
  - `analyzer.py`
  - `midi2corpus.py`
  - `corpus2events.py`
  - `events2musemorphose.py`
  
```
cd custom_data
python3 synchronizer.py
python3 analyzer.py
python3 midi2corpus.py
python3 corpus2events.py
python3 events2musemorphose.py
```

- Step 5:
  - Go to back the parent folder and run `attributes.py` 
```
cd ..
python3 attributes.py
```

- Done! Now you can generate new music by running the following comand:
```
python3 generate.py config/default.yaml musemorphose_pretrained_weights.pt generations/ [num pieces] [num samples per piece]
```
