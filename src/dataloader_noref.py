"""
On-the-fly Q/A generation for an Audio LLM using NISQA Dataset.

Metadata has columns:
['db', 'con', 'file', 'con_description', 'filename_deg', 'filename_ref',
 'source', 'lang', 'votes', 'mos', 'noi', 'col', 'dis', 'loud',
 'noi_std', 'col_std', 'dis_std', 'loud_std', 'mos_std', 'filepath_deg',
 'filepath_ref']
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sig
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, AutoTokenizer

AST_DIR = "MIT/ast-finetuned-audioset-10-10-0.4593"
LLAMA_DIR = "meta-llama/Llama-3.1-8B-Instruct"

TOKENIZER = AutoTokenizer.from_pretrained(LLAMA_DIR)
TOKENIZER.pad_token = TOKENIZER.eos_token

AST_FEATURE_EXTRACTOR = ASTFeatureExtractor.from_pretrained(AST_DIR)


"""
Prompt Template and Feature Extractors
"""


def prompt_template_fn(
    prompt="Evaluate the quality of provided audio.",
    system_message="You are an audio quality evaluation expert.",
):
    prompt_prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {prompt}
    """
    return prompt_prefix


def end_template():
    return """
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """


def bin_score(
    y: float,
    boundaries=(1.5, 2.5, 3.5, 4.5),
    labels=("very bad", "poor", "fair", "good", "excellent"),
) -> str:
    """
    Map a continuous score y ∈ [1,5] to a categorical label.

    Default bins:
      [1,1.5) -> "very bad"
      [1.5,2.5) -> "poor"
      [2.5,3.5) -> "fair"
      [3.5,4.5) -> "good"
      [4.5,5] -> "excellent"
    """
    b1, b2, b3, b4 = boundaries
    if y < b1:
        return labels[0]
    elif y < b2:
        return labels[1]
    elif y < b3:
        return labels[2]
    elif y < b4:
        return labels[3]
    else:
        return labels[4]


@dataclass
class NISQATemplateBank:
    """
    Holds natural language templates and returns random (question, answer) pairs
    for a given row of NISQA metadata.
    """

    def __post_init__(self):
        # Overall MOS numeric templates
        self.q_mos_numeric = [
            (
                "On a scale from 1 (very bad) to 5 (excellent), "
                "what is the overall listening quality of the degraded audio"
                "?"
            ),
            (
                "Rate the overall mean opinion score (MOS) of the degraded speech "
                "using a 1–5 scale."
            ),
            (
                "What overall MOS score from 1 to 5 would you assign "
                "to the degraded audio?"
            ),
        ]

        self.a_mos_numeric = [
            "{mos:.1f}",
            "I would rate the overall MOS as {mos:.1f} out of 5.",
            "Overall MOS ≈ {mos:.1f}.",
        ]

        # Dimension-specific numeric templates
        # dim_name -> (question_templates, answer_templates)
        self.q_dim_numeric = {
            "noi": [
                (
                    "Rate the speech quality in terms of noisiness of the degraded audio "
                    "on a 1–5 scale (higher means less annoying noise)."
                ),
                (
                    "On a 1–5 scale, how would you score the noisiness "
                    "of the degraded signal"
                    "?"
                ),
            ],
            "col": [
                (
                    "Rate the coloration (timbre / bandwidth artifacts) of the "
                    "degraded speech "
                    "on a 1–5 scale "
                    "(higher means more natural timbre)."
                ),
                (
                    "On a 1–5 scale, how is the coloration quality of the "
                    "degraded audio"
                    "?"
                ),
            ],
            "dis": [
                (
                    "Rate the discontinuity (glitches, dropouts) of the degraded audio "
                    "on a 1–5 scale "
                    "(higher means fewer discontinuities)."
                ),
                (
                    "On a 1–5 scale, how smooth is the degraded audio "
                    "in terms of discontinuities"
                    "?"
                ),
            ],
            "loud": [
                (
                    "Rate the loudness quality of the degraded speech "
                    "on a 1–5 scale (higher means more "
                    "comfortable loudness)."
                ),
                (
                    "On a 1–5 scale, how would you score the loudness quality "
                    "of the degraded audio?"
                ),
            ],
        }

        self.a_dim_numeric = {
            dim: [
                "{score:.1f}",
                "I would rate this dimension as {score:.1f} out of 5.",
                "{score:.1f} on the 1–5 scale.",
            ]
            for dim in ["noi", "col", "dis", "loud"]
        }

        # Dimension-specific categorical templates (uses bin_score)
        self.q_dim_categ = {
            "noi": [
                (
                    "How is the degraded audio in terms of background noisiness? "
                    "Use categories: very bad, poor, fair, good, or excellent."
                ),
                (
                    "Describe the noisiness quality of the degraded speech: "
                    "is it very bad, poor, fair, good, or excellent?"
                ),
            ],
            "col": [
                (
                    "How would you describe the coloration (timbre) of the "
                    "degraded speech? "
                    "Choose from very bad, poor, fair, good, or excellent."
                ),
            ],
            "dis": [
                (
                    "How would you describe temporal discontinuities in the "
                    "degraded audio (clicks, dropouts, glitches)? "
                    "Use very bad, poor, fair, good, or excellent."
                ),
            ],
            "loud": [
                (
                    "How is the loudness quality of the degraded audio? "
                    "Is the level very bad, poor, fair, good, or excellent?"
                ),
            ],
        }

        self.a_dim_categ = {
            "noi": [
                "The noisiness quality is {cat}, about {score:.1f} out of 5.",
                "I would say the noisiness is {cat} (≈{score:.1f}/5).",
            ],
            "col": [
                "The coloration is {cat}, around {score:.1f} on the scale.",
            ],
            "dis": [
                "The discontinuity quality is {cat}, about {score:.1f}/5.",
            ],
            "loud": [
                "The loudness quality is {cat}, roughly {score:.1f}/5.",
            ],
        }

        # Multi-dimensional summary (all 5 scores)
        self.q_multi_dim = [
            (
                "Provide a quality assessment of the degraded speech"
                ", including overall MOS and the four "
                "dimensions: noisiness, coloration, discontinuity, and "
                "loudness. Give scores between 1 and 5."
            ),
            (
                "Summarize the quality of the degraded audio "
                "by providing numeric scores (1–5) for: overall MOS, "
                "noisiness, coloration, discontinuity, and loudness quality."
            ),
        ]

        self.a_multi_dim = [
            (
                "Overall MOS: {mos:.1f}. "
                "Noisiness: {noi:.1f}. "
                "Coloration: {col:.1f}. "
                "Discontinuity: {dis:.1f}. "
                "Loudness quality: {loud:.1f}."
            ),
            (
                "I would assign the following scores (1–5): "
                "overall MOS = {mos:.1f}, "
                "noisiness = {noi:.1f}, "
                "coloration = {col:.1f}, "
                "discontinuity = {dis:.1f}, "
                "loudness = {loud:.1f}."
            ),
        ]

        self.q_explanatory = [
            (
                "Explain the main causes of quality degradation in the "
                "degraded speech, then provide "
                "an overall MOS score between 1 and 5."
            )
        ]

        self.a_explanatory = [
            (
                "The degraded audio suffers from {cause_desc}. "
                "Considering the combined effects on noisiness (≈{noi:.1f}), "
                "coloration (≈{col:.1f}), discontinuity (≈{dis:.1f}), and "
                "loudness quality (≈{loud:.1f}), I would give an overall MOS "
                "of {mos:.1f}."
            )
        ]

    def generate_qa(
        self,
        row: Dict[str, Any],
        rng: random.Random,
        allowed_tasks=("mos_numeric", "dim_numeric", "dim_categ", "multi_dim"),
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate a (question, answer, meta) triple for a given NISQA row.

        row: dict with at least keys: mos, noi, col, dis, loud
        rng: random.Random instance (caller controls seeding)
        allowed_tasks: subset of {"mos_numeric", "dim_numeric",
                                  "dim_categ", "multi_dim", "explanatory"}

        Returns:
            question: str
            answer: str
            meta: dict with fields like {"task": "...", "dim": "..."} for bookkeeping
        """
        # Extract numeric labels
        mos = float(row["mos"])
        noi = float(row["noi"])
        col = float(row["col"])
        dis = float(row["dis"])
        loud = float(row["loud"])

        con_desc = row.get("con_description", "")

        # Choose a task family
        task = rng.choice(list(allowed_tasks))

        if task == "mos_numeric":
            q = rng.choice(self.q_mos_numeric)
            a = rng.choice(self.a_mos_numeric).format(mos=mos)
            meta = {"task": "mos_numeric"}

        elif task == "dim_numeric":
            dim = rng.choice(["noi", "col", "dis", "loud"])
            score = {"noi": noi, "col": col, "dis": dis, "loud": loud}[dim]
            q = rng.choice(self.q_dim_numeric[dim])
            a = rng.choice(self.a_dim_numeric[dim]).format(score=score)
            meta = {"task": "dim_numeric", "dim": dim}

        elif task == "dim_categ":
            dim = rng.choice(["noi", "col", "dis", "loud"])
            score = {"noi": noi, "col": col, "dis": dis, "loud": loud}[dim]
            cat = bin_score(score)
            q = rng.choice(self.q_dim_categ[dim])
            a = rng.choice(self.a_dim_categ[dim]).format(score=score, cat=cat)
            meta = {"task": "dim_categ", "dim": dim}

        elif task == "multi_dim":
            q = rng.choice(self.q_multi_dim)
            a = rng.choice(self.a_multi_dim).format(
                mos=mos, noi=noi, col=col, dis=dis, loud=loud
            )
            meta = {"task": "multi_dim"}

        elif task == "explanatory":
            cause_desc = (
                con_desc if con_desc else "typical network and noise distortions"
            )
            q = rng.choice(self.q_explanatory)
            a = rng.choice(self.a_explanatory).format(
                cause_desc=cause_desc,
                mos=mos,
                noi=noi,
                col=col,
                dis=dis,
                loud=loud,
            )
            meta = {"task": "explanatory"}

        else:
            raise ValueError(f"Unknown task type: {task}")

        return q, a, meta


def load_wav_mono(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load a mono waveform from `path` using soundfile.

    Returns:
        wav: np.ndarray, shape (T,)
        sr: int
    """
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return np.zeros(0, dtype=np.float32), -1

    wav, sr = sf.read(path)  # wav: (T,) or (T, C)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # downmix to mono

    wav = wav.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        wav = sig.resample(wav, int(len(wav) * (target_sr / sr)))

        return wav, target_sr

    return wav, sr


class NISQAAudioQADataset(Dataset):
    """
    PyTorch Dataset for Audio LLM:

      - reads rows from a NISQA CSV
      - loads degraded audio
      - generates (question, answer) text on the fly

    It returns a dictionary with:
      {
        "audio_deg": torch.FloatTensor (T_deg,),
        "sr_deg": int,
        "question": str,
        "answer": str,
        "labels": { "mos":..., "noi":..., "col":..., "dis":..., "loud":... },
        "meta": {...}
      }
    """

    def __init__(
        self,
        csv_path: str,
        dataset_split: str = "TRAIN",
        rng_seed: int = 0,
        allowed_tasks=("mos_numeric", "dim_numeric", "dim_categ", "multi_dim"),
        target_duration: float = 10.0,
        target_sr: int = 16000,
    ):
        """
        csv_path: path to NISQA-style CSV
        rng_seed: base seed for internal RNG
        allowed_tasks: which task families to use in generate_qa
        target_sr: resample audio to this rate in load_wav_mono
        """
        super().__init__()

        self.target_sr = target_sr
        self.target_duration = target_duration

        dataset = pd.read_csv(csv_path)
        self.table = dataset[dataset["db"].str.contains(dataset_split)]

        self.allowed_tasks = allowed_tasks
        self.target_sr = target_sr

        self.templates = NISQATemplateBank()
        self.rng = random.Random(rng_seed)

        # Cache columns as numpy arrays for faster indexing
        self._rows = self.table.to_dict("records")

    def __len__(self) -> int:
        return len(self._rows)

    def _get_rng(self) -> random.Random:
        """
        Return an RNG.
        """
        return self.rng

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._rows[idx]
        rng = self._get_rng()

        # 1) Generate question/answer from labels
        question, answer, qa_meta = self.templates.generate_qa(
            row, rng, allowed_tasks=self.allowed_tasks
        )

        # 2) Load degraded audio

        path_deg = "Dataset/NISQA_Corpus/" + row["filepath_deg"]

        wav_deg, sr_deg = load_wav_mono(path_deg, target_sr=self.target_sr)

        # 3) Create a window of 8 seconds
        target_len = int(self.target_duration * sr_deg)  # number of samples for 8 s

        if wav_deg.shape[0] >= target_len:
            # random crop
            start = np.random.randint(0, wav_deg.shape[0] - target_len + 1)
            wav_deg = wav_deg[start : start + target_len]
        else:
            # pad with zeros at the end
            pad_len = target_len - wav_deg.shape[0]
            wav_deg = np.pad(wav_deg, (0, pad_len), mode="constant")

        # 4) Extract Audio Feature
        feature_deg = AST_FEATURE_EXTRACTOR(wav_deg, sampling_rate=self.target_sr)[
            "input_values"
        ][0]

        # 5) Prepare numeric labels
        labels = {
            "mos": float(row["mos"]),
            "noi": float(row["noi"]),
            "col": float(row["col"]),
            "dis": float(row["dis"]),
            "loud": float(row["loud"]),
            "mos_std": float(row.get("mos_std", np.nan)),
            "noi_std": float(row.get("noi_std", np.nan)),
            "col_std": float(row.get("col_std", np.nan)),
            "dis_std": float(row.get("dis_std", np.nan)),
            "loud_std": float(row.get("loud_std", np.nan)),
            "qa_task": qa_meta.get("task", ""),
            "qa_dim": qa_meta.get("dim", None),
        }

        # 6) Prepare Tokens
        sq_tokens = TOKENIZER(answer, padding=True, truncation=True)
        speech_quality_ids = sq_tokens.input_ids
        speech_quality_mask = sq_tokens.attention_mask

        prompt_tokens = TOKENIZER(
            prompt_template_fn(question), padding=True, truncation=True
        )
        prompt_ids = prompt_tokens.input_ids
        prompt_mask = prompt_tokens.attention_mask

        end_prompt_tokens = TOKENIZER(end_template(), padding=True, truncation=True)
        end_prompt_ids = end_prompt_tokens.input_ids
        end_prompt_mask = end_prompt_tokens.attention_mask

        return {
            "sr_deg": sr_deg,
            "question": question,
            "answer": answer,
            "labels": labels,
            "noisy_features": feature_deg,
            "speech_quality_ids": speech_quality_ids,
            "speech_quality_attention_mask": speech_quality_mask,
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "end_prompt_ids": end_prompt_ids,
            "end_prompt_attention_mask": end_prompt_mask,
        }


if __name__ == "__main__":
    # Example: constructing the dataset
    csv_path = "Dataset/NISQA_Corpus/NISQA_corpus_file.csv"  # CSV path here

    dataset = NISQAAudioQADataset(
        csv_path=csv_path,
        dataset_split="TEST",
        rng_seed=1234,
        allowed_tasks=("mos_numeric", "dim_numeric", "dim_categ", "multi_dim"),
        target_sr=16000,
        target_duration=10.0,
    )
    print(len(dataset))

    # Inspect one example
    ex = dataset[0]

    print("Question:")
    print(ex["question"])
    print()
    print("Answer:")
    print(ex["answer"])
    print()
    print("Labels:", ex["labels"])
    print("audio_deg shape:", ex["noisy_features"].shape, "sr_deg:", ex["sr_deg"])
