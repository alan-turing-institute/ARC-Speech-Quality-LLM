"""
Evaluate a pre-trained no-reference SpeechQualityLLM on a directory of TTS model
outputs and produce per-model average scores for ranking.

Expected directory layout:
    root_dir/
        model_A/
            clip1.wav
            clip2.wav
        model_B/
            clip1.wav
            ...

For each .wav file, the script runs the model on three task types
(mos_numeric, dim_numeric, dim_categ) and collects predicted scores.
It then averages scores per TTS model to produce a ranking.
"""

import argparse
import os
import re
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.dataloader_noref import (
    AST_DIR,
    AST_FEATURE_EXTRACTOR,
    LLAMA_DIR,
    TOKENIZER,
    end_template,
    load_wav_mono,
    prompt_template_fn,
)

if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic question templates (first template per task/dim from
# NISQATemplateBank — fixed for reproducibility)
# ──────────────────────────────────────────────────────────────────────────────

QUESTION_TEMPLATES = {
    "mos_numeric": {
        "mos": (
            "On a scale from 1 (very bad) to 5 (excellent), "
            "what is the overall listening quality of the degraded audio?"
        ),
    },
    "dim_numeric": {
        "noi": (
            "Rate the speech quality in terms of noisiness of the degraded audio "
            "on a 1–5 scale (higher means less annoying noise)."
        ),
        "col": (
            "Rate the coloration (timbre / bandwidth artifacts) of the "
            "degraded speech on a 1–5 scale "
            "(higher means more natural timbre)."
        ),
        "dis": (
            "Rate the discontinuity (glitches, dropouts) of the degraded audio "
            "on a 1–5 scale "
            "(higher means fewer discontinuities)."
        ),
        "loud": (
            "Rate the loudness quality of the degraded speech "
            "on a 1–5 scale (higher means more "
            "comfortable loudness)."
        ),
    },
    "dim_categ": {
        "noi": (
            "How is the degraded audio in terms of background noisiness? "
            "Use categories: very bad, poor, fair, good, or excellent."
        ),
        "col": (
            "How would you describe the coloration (timbre) of the "
            "degraded speech? "
            "Choose from very bad, poor, fair, good, or excellent."
        ),
        "dis": (
            "How would you describe temporal discontinuities in the "
            "degraded audio (clicks, dropouts, glitches)? "
            "Use very bad, poor, fair, good, or excellent."
        ),
        "loud": (
            "How is the loudness quality of the degraded audio? "
            "Is the level very bad, poor, fair, good, or excellent?"
        ),
    },
}

DUMMY_ANSWER = "The audio quality is: "

# ──────────────────────────────────────────────────────────────────────────────
# Model (same architecture as evaluate_noref.py)
# ──────────────────────────────────────────────────────────────────────────────


class AudioProjectionLayer(nn.Module):
    def __init__(
        self,
        input_embedding_size=768,
        output_embedding_size=4096,
        temporal_pooling=128,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(temporal_pooling)
        self.layer_norm = nn.LayerNorm(input_embedding_size)
        self.linear_projection = nn.Linear(
            in_features=input_embedding_size,
            out_features=output_embedding_size,
        )

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.pool(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = self.linear_projection(x)
        return x


class SpeechQualityLLM(nn.Module):
    def __init__(self, ast_encoder, llm, pooling_length=128):
        super().__init__()
        self.audio_encoder = ast_encoder

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora_cfg)

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
            if "attention.query" in name:
                param.requires_grad = True

        self.audio_projection_layer = AudioProjectionLayer(
            input_embedding_size=768,
            output_embedding_size=4096,
            temporal_pooling=pooling_length,
        )

    def forward_without_llm(
        self,
        noisy_features,
        prompt_ids,
        prompt_attention_mask,
        end_prompt_ids,
        end_prompt_attention_mask,
        speech_quality_ids,
        speech_quality_attention_mask,
    ):
        noisy_tokens = self.audio_encoder(noisy_features).last_hidden_state
        projected_noisy_audio = self.audio_projection_layer(noisy_tokens)

        tok_emb = self.llm.get_input_embeddings()
        speech_quality_embeds = tok_emb(speech_quality_ids)
        prompt_embeds = tok_emb(prompt_ids)
        end_prompt_embeds = tok_emb(end_prompt_ids)

        bs, noisy_audio_token_len = projected_noisy_audio.shape[:2]

        query_embeds = torch.cat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
            ),
            dim=1,
        )
        query_attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return query_embeds, query_attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_modules():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR,
        quantization_config=bnb_config,
    )
    ast_encoder = AutoModelForAudioClassification.from_pretrained(
        AST_DIR
    ).audio_spectrogram_transformer
    return llm, ast_encoder


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    if os.path.isdir(checkpoint_path):
        st_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.isfile(st_file):
            raise FileNotFoundError(
                f"No 'model.safetensors' found in directory {checkpoint_path}"
            )
    else:
        st_file = checkpoint_path

    print(f"Loading checkpoint weights from {st_file}")

    if st_file.endswith(".safetensors"):
        state_dict = load_file(st_file, device="cpu")
    else:
        state_dict = torch.load(st_file, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


# ──────────────────────────────────────────────────────────────────────────────
# Directory scanning
# ──────────────────────────────────────────────────────────────────────────────


def discover_tts_models(root_dir: str) -> Dict[str, List[str]]:
    """
    Scan root_dir for subdirectories, each treated as a TTS model.
    Returns {model_name: [wav_path, ...]}.
    """
    models = {}
    for entry in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        wav_files = sorted(glob(os.path.join(model_dir, "*.wav")))
        if wav_files:
            models[entry] = wav_files
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class TTSAudioDataset(Dataset):
    """
    PyTorch Dataset for running a single (task, dim) evaluation across
    a list of (model_name, wav_path) pairs.
    """

    def __init__(
        self,
        audio_files: List[Tuple[str, str]],
        task: str,
        dim: str,
        target_sr: int = 16000,
        target_duration: float = 10.0,
    ):
        self.audio_files = audio_files
        self.task = task
        self.dim = dim
        self.target_sr = target_sr
        self.target_duration = target_duration

        self.question = QUESTION_TEMPLATES[task][dim]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        model_name, wav_path = self.audio_files[idx]

        # Load and window audio
        wav, _sr = load_wav_mono(wav_path, target_sr=self.target_sr)
        target_len = int(self.target_duration * self.target_sr)
        if wav.shape[0] >= target_len:
            wav = wav[:target_len]
        else:
            pad_len = target_len - wav.shape[0]
            wav = np.pad(wav, (0, pad_len), mode="constant")

        # AST features
        feature = AST_FEATURE_EXTRACTOR(wav, sampling_rate=self.target_sr)[
            "input_values"
        ][0]

        # Tokenise
        prompt_tokens = TOKENIZER(
            prompt_template_fn(self.question), padding=True, truncation=True
        )
        end_prompt_tokens = TOKENIZER(end_template(), padding=True, truncation=True)
        sq_tokens = TOKENIZER(DUMMY_ANSWER, padding=True, truncation=True)

        return {
            "noisy_features": feature,
            "prompt_ids": prompt_tokens.input_ids,
            "prompt_attention_mask": prompt_tokens.attention_mask,
            "end_prompt_ids": end_prompt_tokens.input_ids,
            "end_prompt_attention_mask": end_prompt_tokens.attention_mask,
            "speech_quality_ids": sq_tokens.input_ids,
            "speech_quality_attention_mask": sq_tokens.attention_mask,
            "model_name": model_name,
            "file_path": wav_path,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate / helpers
# ──────────────────────────────────────────────────────────────────────────────


def pad_sequence_start(sequences, batch_first=False, padding_value=0.0):
    max_length = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_length = max_length - seq.size(0)
        padding = torch.full(
            (pad_length,) + seq.size()[1:],
            padding_value,
            dtype=seq.dtype,
            device=seq.device,
        )
        padded_seqs.append(torch.cat([seq, padding], dim=0))
    if batch_first:
        return torch.stack(padded_seqs, dim=0)
    else:
        return torch.stack(padded_seqs, dim=1)


def collate_fn(batch):
    noisy_feature = [
        AST_FEATURE_EXTRACTOR.pad(
            {"input_values": item["noisy_features"]}, return_tensors="pt"
        ).input_values
        for item in batch
    ]
    noisy_features = torch.stack(noisy_feature)

    def _pad_ids(key, pad_val):
        return pad_sequence_start(
            [torch.tensor(item[key]) for item in batch],
            batch_first=True,
            padding_value=pad_val,
        )

    return {
        "noisy_features": noisy_features,
        "prompt_ids": _pad_ids("prompt_ids", TOKENIZER.pad_token_id),
        "prompt_attention_mask": _pad_ids("prompt_attention_mask", 0),
        "end_prompt_ids": _pad_ids("end_prompt_ids", TOKENIZER.pad_token_id),
        "end_prompt_attention_mask": _pad_ids("end_prompt_attention_mask", 0),
        "speech_quality_ids": _pad_ids("speech_quality_ids", TOKENIZER.pad_token_id),
        "speech_quality_attention_mask": _pad_ids("speech_quality_attention_mask", 0),
        "model_names": [item["model_name"] for item in batch],
        "file_paths": [item["file_path"] for item in batch],
    }


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Float extraction (same as evaluate_noref.py)
# ──────────────────────────────────────────────────────────────────────────────

_float_regex = re.compile(r"[-+]?\d*\.?\d+")


def extract_floats(text: str, min_val=0.0, max_val=6.0):
    nums = []
    for m in _float_regex.findall(text):
        try:
            x = float(m)
            if min_val <= x <= max_val:
                nums.append(x)
        except ValueError:
            continue
    return nums


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────


def run_task_inference(
    task: str,
    dim: str,
    model: SpeechQualityLLM,
    loader: DataLoader,
    device: torch.device,
    no_temperature: bool = False,
) -> List[dict]:
    """
    Run inference for a single (task, dim) and return per-file prediction rows.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{task}/{dim}"):
            model_names = batch.pop("model_names")
            file_paths = batch.pop("file_paths")

            batch = move_to_device(batch, device)

            query_embeds, query_attention_mask = model.forward_without_llm(**batch)

            gen_ids = model.llm.generate(
                inputs_embeds=query_embeds,
                attention_mask=query_attention_mask,
                max_length=query_embeds.shape[1] + 64,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
                do_sample=not no_temperature,
            )
            generated = TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)

            for mname, fpath, gen_text in zip(model_names, file_paths, generated):
                floats = extract_floats(gen_text, min_val=0.0, max_val=6.0)
                pred = floats[0] if floats else None
                rows.append(
                    {
                        "model_name": mname,
                        "file_path": fpath,
                        "task": task,
                        "dim": dim,
                        "predicted_score": pred,
                        "generated_text": gen_text,
                    }
                )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────


def aggregate_scores(
    all_rows: List[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From per-file prediction rows, produce:
      1) predictions_df: per-file predictions (for CSV)
      2) summary_df: per-model average scores with ranking
    """
    predictions_df = pd.DataFrame(all_rows)

    # Build per-model summary
    valid = predictions_df.dropna(subset=["predicted_score"])

    # Create a column key like "mos_numeric_mos", "dim_numeric_noi", etc.
    valid = valid.copy()
    valid["score_key"] = valid["task"] + "_" + valid["dim"]

    # Pivot: mean score per model per score_key
    pivot = valid.pivot_table(
        index="model_name",
        columns="score_key",
        values="predicted_score",
        aggfunc="mean",
    )

    # Count successful predictions per model
    counts = valid.pivot_table(
        index="model_name",
        columns="score_key",
        values="predicted_score",
        aggfunc="count",
    )
    counts.columns = [f"{c}_n" for c in counts.columns]

    summary_df = pivot.join(counts)

    # Sort by MOS descending if available
    if "mos_numeric_mos" in summary_df.columns:
        summary_df = summary_df.sort_values("mos_numeric_mos", ascending=False)

    summary_df.index.name = "model_name"
    summary_df = summary_df.reset_index()

    return predictions_df, summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rank TTS models by predicted speech quality scores."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing one subdirectory per TTS model, each with .wav files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/NoReference_FrozenAST/checkpoint-10240",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--target_duration", type=float, default=10.0)
    parser.add_argument(
        "--tasks",
        type=str,
        default="mos_numeric,dim_numeric,dim_categ",
        help="Comma-separated list of tasks to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/tts_ranking",
    )
    parser.add_argument(
        "--no_temperature",
        type=bool,
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover TTS models
    tts_models = discover_tts_models(args.root_dir)
    if not tts_models:
        print(f"No TTS model directories with .wav files found in {args.root_dir}")
        return

    total_files = sum(len(v) for v in tts_models.values())
    print(f"Found {len(tts_models)} TTS models, {total_files} total .wav files:")
    for mname, files in tts_models.items():
        print(f"  {mname}: {len(files)} files")

    # Build flat list of (model_name, wav_path) for dataset construction
    audio_files = []
    for model_name, wav_paths in tts_models.items():
        for wp in wav_paths:
            audio_files.append((model_name, wp))

    # Load model
    print("\nLoading model...")
    llm_base, ast_encoder = load_modules()
    model = SpeechQualityLLM(ast_encoder=ast_encoder, llm=llm_base, pooling_length=128)
    model = model.to(device)
    load_checkpoint(model, args.checkpoint_path)
    model.llm.config.pad_token_id = TOKENIZER.pad_token_id

    # Run inference for each (task, dim)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_rows = []

    for task in task_list:
        if task not in QUESTION_TEMPLATES:
            print(f"Warning: unknown task '{task}', skipping.")
            continue

        dims = list(QUESTION_TEMPLATES[task].keys())
        for dim in dims:
            print(f"\n=== Running inference: task={task}, dim={dim} ===")

            dataset = TTSAudioDataset(
                audio_files=audio_files,
                task=task,
                dim=dim,
                target_sr=args.target_sr,
                target_duration=args.target_duration,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            rows = run_task_inference(
                task=task,
                dim=dim,
                model=model,
                loader=loader,
                device=device,
                no_temperature=args.no_temperature,
            )
            all_rows.extend(rows)

            # Quick summary
            scores = [
                r["predicted_score"] for r in rows if r["predicted_score"] is not None
            ]
            n_fail = sum(1 for r in rows if r["predicted_score"] is None)
            if scores:
                print(
                    f"  Predictions: {len(scores)}/{len(rows)} "
                    f"(failed: {n_fail}), "
                    f"mean={np.mean(scores):.2f}, "
                    f"std={np.std(scores):.2f}"
                )
            else:
                print(f"  No valid predictions extracted ({n_fail} failures)")

    # Aggregate and save
    predictions_df, summary_df = aggregate_scores(all_rows)

    preds_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(preds_path, index=False)
    print(f"\nSaved per-file predictions to {preds_path}")

    summary_csv_path = os.path.join(args.output_dir, "ranking_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved ranking summary to {summary_csv_path}")

    summary_json_path = os.path.join(args.output_dir, "ranking_summary.json")
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    print(f"Saved ranking summary to {summary_json_path}")

    # Print ranking table
    print("\n" + "=" * 80)
    print("TTS MODEL RANKING")
    print("=" * 80)

    # Identify score columns (exclude _n count columns and model_name)
    score_cols = [
        c for c in summary_df.columns if c != "model_name" and not c.endswith("_n")
    ]

    # Header
    header = f"{'Rank':<5} {'Model':<30}"
    for col in score_cols:
        short = (
            col.replace("mos_numeric_", "")
            .replace("dim_numeric_", "")
            .replace("dim_categ_", "categ_")
        )
        header += f" {short:>10}"
    print(header)
    print("-" * len(header))

    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        line = f"{rank:<5} {row['model_name']:<30}"
        for col in score_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val:>10.2f}"
            else:
                line += f" {'N/A':>10}"
        print(line)

    print("=" * 80)


if __name__ == "__main__":
    main()
