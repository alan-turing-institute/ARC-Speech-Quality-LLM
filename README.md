# SpeechQualityLLM: LLM-based Interactive Assessment of Speech Quality

This is a fork of the [SpeechQualityLLM repo](https://github.com/Monjur-Mahathir/Speech-Quality-LLM?tab=readme-ov-file), updated to be `uv` installable so that it can be run in 2026. It now interfaces directly with huggingface to download and cache the base models.

> **TL;DR.** SpeechQualityLLM turns objective speech quality assessment into a **question–answering task**:  
> given a (degraded, optional reference) speech signal and a natural-language question, a multimodal LLM predicts
> MOS and dimension-wise scores **and** explains its reasoning in text.

---

## Installation
If `uv` is not installed it can be installed from the following command:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Following this to install SpeechQualityLLM run,

```
uv sync 
```

---

## Overview

This repository contains code for **SpeechQualityLLM**, a multimodal system that:

- Takes **degraded speech** (and optionally a **clean reference**) as input.
- Uses an **audio encoder** (AST or Whisper) to extract time–frequency representations.
- Projects audio features into a **token sequence** and feeds them, along with a textual prompt, into a **LLaMA-family language model**.
- Answers speech-quality questions in natural language, including:
  - **MOS-numeric** (overall 1–5 MOS),
  - **Dim-numeric** (noisiness, coloration, discontinuity, loudness),
  - **Dim-categorical** (verbal ratings),
  - **Multi-dim** (joint MOS + four dimensions),
  - **Explanatory** (MOS with short rationale).

The model is trained on the **NISQA** dataset using automatically generated question–answer pairs, and is evaluated on both single-ended (no-reference) and full-reference settings.

---

## Main Features

- **End-to-end multimodal QA** for speech quality (audio + text → text).
- Support for **single-ended** and **double-ended** quality assessment.
- Multiple audio backbones:
  - [AST](https://arxiv.org/abs/2104.01778) (Audio Spectrogram Transformer)
  - [Whisper](https://openai.com/research/whisper) encoder
- **LoRA-tuned** LLaMA backbone (e.g., LLaMA 3.1 8B) with 4-bit base weights.
- Rich **textual outputs**: scores, rationales, and profile-conditioned prompts
  (“act like a very noise-sensitive listener”, etc.).
- Reproducible evaluation on NISQA:
  - MOS / dimension MAE, RMSE
  - Pearson / Spearman correlations
  - Multi-dimension consistency
- Ranking of TTS models by the above speech quality metrics given adequate samples from each.

---

## Repository Structure

```text
Speech-Quality-LLM/
├─ README.md
├─ LICENSE
├─ pyproject.toml                  # Python dependencies
├─ uv.lock
├─ Dataset/
│   └─ NISQA_Corpus                  # NISQA metadata, audio files etc. (not included)
├─ results/
│   └─ Reference
        └─ checkpoint-10240          # Checkpoints after training of 10,000 steps, downloadable from google drive: https://drive.google.com/drive/folders/1vzcmHgOIpqVe6KzQBUfI5lHOd4slUREO?usp=sharing
    └─ NoReference
        └─ checkpoint-10240
    ....
├─ src/
│   ├─ dataloader_noref.py
│   ├─ dataloader_ref.py
│   ├─ dataloader_ref_whisper.py
│   ├─ training_noref.py
│   ├─ training_ref.py
│   ├─ training_ref_whisper.py
├─ eval_scripts/
│   ├─ evaluate_noref.py
│   ├─ evaluate_ref_whisper.py
│   ├─ evaluate_ref.py
│   ├─ evaluate.py
└─ train.py
```


## Evaluation
For repeatability, append the following with `--no_temperature True`
1. In order to run the Full-reference model with finetuned AST, please run the evaluate.py with the necessary degraded audio path, reference audio path, prompt arguments, as well as checkpoint path if needed. Example:
```text
  python evaluate.py --deg_path Dataset/NISQA_Corpus/NISQA_TEST_FOR/deg/c00001_for_cnv_m_1035_02.wav --ref_path Dataset/NISQA_Corpus/NISQA_TEST_FOR/ref/for_cnv_m_1035_02.wav --prompt "Explain the main causes of quality degradation in the degraded speech compared to the reference, then provide an overall MOS score between 1 and 5."
```
2. In order to evaluate the models on NISQA test dataset, run the following scripts:
```text
  python evaluate_ref.py --checkpoint_path results/Reference/checkpoint-10240 # (for Full-reference with AST encoder)
  python evaluate_noref.py --checkpoint_path results/NoReference/checkpoint-10240 # (for No-reference with AST encoder)
python evaluate_ref_whisper.py --checkpoint_path results/Reference_FrozenWhisper/checkpoint-10240 # (for Full-reference with Whisper encoder)
```
## Training
In order to train the model on NISQA train, validation and test set, please run the following script:
```text
  python train.py --training_type full_reference_ast --freeze_encoder False 
```
Use any of ["full_reference_ast", "no_reference_ast", "full_reference_whisper"] training type and keep the audio encoder frozen or trainable (by default finetune query projection layers only). Also change the dataset path and training params as needed.
## Citation
If you find this repository useful in your research, please consider citing the following work:
```text
@misc{monjur2025speechqualityllmllmbasedmultimodalassessment,
      title={SpeechQualityLLM: LLM-Based Multimodal Assessment of Speech Quality}, 
      author={Mahathir Monjur and Shahriar Nirjon},
      year={2025},
      eprint={2512.08238},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.08238}, 
}
```
