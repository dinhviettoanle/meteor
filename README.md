# METEOR: Melody-aware Texture-controllable Symbolic Orchestral Music Generation via Transformer VAE

[![Demo](https://img.shields.io/badge/🎵-Demo-blue)](https://dinhviettoanle.github.io/meteor/) [![arXiv Badge](https://img.shields.io/badge/arXiv-2409.11753-B31B1B?logo=arxiv&logoColor=fff&style=flat)](
http://arxiv.org/abs/2409.11753) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/dinhviettoanle/meteor) 

<!-- *Accepted to the 34th International Joint Conference on Artificial Intelligence (IJCAI), Special Track on AI, Arts and Creativity,*

*Dinh-Viet-Toan Le, Yi-Hsuan Yang* -->

**Abstract**
Re-orchestration is the process of adapting a music piece for a different set of instruments. By altering the original instrumentation, the orchestrator often modifies the musical texture while preserving a recognizable melodic line and ensures that each part is playable within the technical and expressive capabilities of the chosen instruments.
In this work, we propose METEOR, a model for generating **Me**lody-aware **Te**xture-controllable re-**Or**chestration with a Transformer-based variational auto-encoder (VAE). This model performs symbolic instrumental and textural music style transfers with a focus on melodic fidelity and controllability. We allow bar- and track-level controllability of the accompaniment with various textural attributes while keeping a homophonic texture. With both subjective and objective evaluations, we show that our model outperforms style transfer models on a re-orchestration task in terms of generation quality and controllability. Moreover, it can be adapted for a lead sheet orchestration task as a zero-shot learning model, achieving performance comparable to a model specifically trained for this task. 

---

## Setup
Create environment:
```
conda create -n envmeteor python=3.9.2
conda activate envmeteor
```

Install requirements:
```
pip install --no-deps -r requirements.txt
```

## Quickstart: inference from a trained model given a MIDI file
1. Download the checkpoint on [HuggingFace Hub](https://huggingface.co/dinhviettoanle/meteor).
2. Move the files in `models/`.
3. (If GPU available) In `config/demo.yaml`, change `device: cpu` into `device: cuda:0`.
4. Run `sh generate.sh`.

---

## Pre-process custom data
```
python -m src.custom_data.remi_pipeline --config=data.yaml [options]
```

With the following `[options]`:
- `--all`: run the full pipeline (see below).
- `--analyzer`: run the `midi → midi_analyzed` part of the pipeline.
- `--mid2musemorphose`: run the `midi_analyzed → remi+attributes` part of the pipeline.
- `--attributes`: run the `+attributes` part of the pipeline.

In details, the pre-processing steps involve the following formats and folders: `midi → midi_analyzed → corpus → events → musemorphose (or remi) → remi+attributes` 

## Train model
```
python -m src.train config/test.yaml
```

Documentation regarding each field of the `yaml` file is available in `config/documentation.yaml`.

## Generate
```
sh generate.sh
```

Or
```
python -m src.generate\
    config/demo.yaml\
    <path-to-yaml-model-checkpoint>\
    <output-path>\
    <number-pieces>\
    <number-samples-per-piece>\
    [options]
```
With the following `[options]`:
- `--analyze`: analyze the generated afterwards in terms of rhythmicity and polyphonicity.
- `--random_pitchrange`: bar-wise pitch diversity tokens are set randomly.
- `--random_repeatability`: bar-wise pitch diversity tokens are set randomly.


## Citation BibTex
If you find this work helpful and use our code in your research, please cite our paper:
```
@inproceedings{le2025meteor,
  title = {{METEOR}: Melody-aware Texture-controllable Symbolic Music Re-Orchestration via Transformer VAE},
  author = {Le, Dinh-Viet-Toan and Yang, Yi-Hsuan},
  booktitle = {Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI), Special Track on AI, Arts and Creativity},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages = {},
  year = {2025},
  month = aug,
  location = {Montreal, Canada},
  url = {https://arxiv.org/abs/2409.11753},
  note = {[Accepted for publication]}
}
```

This code is based on these repositories:
- [MuseMorphose](https://github.com/YatingMusic/MuseMorphose) for model implementation, training and inference
- [MuseMorphose fork](https://github.com/stavaser/MuseMorphose) and [Compound Word Transformer](https://github.com/YatingMusic/compound-word-transformer) for custom data processing