import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from transformers import (
    WhisperModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    WhisperProcessor
)
from peft import LoraConfig, get_peft_model

from src.dataloader_ref_whisper import (
    NISQAAudioQADataset,
    WHISPER_DIR,
    LLAMA_DIR,
    TOKENIZER,
    WHISPER_FEATURE_EXTRACTOR,
)

if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token


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
        # x: [B, T, C_in]
        x = x.transpose(-2, -1)          # [B, C_in, T]
        x = self.pool(x)                 # [B, C_in, T_p]
        x = x.transpose(-2, -1)          # [B, T_p, C_in]
        x = self.layer_norm(x)
        x = self.linear_projection(x)    # [B, T_p, C_out]
        return x


class SpeechQualityLLM(nn.Module):
    def __init__(self, whisper_encoder, llm, pooling_length=128):
        super().__init__()
        self.audio_encoder = whisper_encoder

        # LoRA cfg must match training
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora_cfg)

        self.audio_projection_layer = AudioProjectionLayer(
            input_embedding_size=1280,
            output_embedding_size=4096,
            temporal_pooling=pooling_length,
        )

    def forward(
        self,
        noisy_features,
        reference_features,
        prompt_ids,
        prompt_attention_mask,
        end_prompt_ids,
        end_prompt_attention_mask,
        speech_quality_ids,
        speech_quality_attention_mask,
    ):
        """
        Training forward (not actually used in eval script, kept for completeness).
        """
        noisy_tokens = self.audio_encoder(noisy_features).last_hidden_state
        projected_noisy_audio = self.audio_projection_layer(noisy_tokens)

        reference_tokens = self.audio_encoder(reference_features).last_hidden_state
        projected_reference_audio = self.audio_projection_layer(reference_tokens)

        tok_emb = self.llm.get_input_embeddings()
        speech_quality_embeds = tok_emb(speech_quality_ids)
        prompt_embeds = tok_emb(prompt_ids)
        end_prompt_embeds = tok_emb(end_prompt_ids)

        bs, noisy_audio_token_len = projected_noisy_audio.shape[:2]
        _, reference_audio_token_len = projected_reference_audio.shape[:2]

        input_embeds = torch.cat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                projected_reference_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
                speech_quality_embeds,
            ),
            dim=1,
        )

        attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                torch.ones(bs, reference_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
                speech_quality_attention_mask,
            ),
            dim=1,
        )

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        total_audio_tokens = noisy_audio_token_len + reference_audio_token_len
        query_embeds = torch.cat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                projected_reference_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
            ),
            dim=1,
        )
        query_attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                torch.ones(bs, reference_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return outputs, query_embeds, query_attention_mask, total_audio_tokens

    def forward_without_llm(
        self,
        noisy_features,
        reference_features,
        prompt_ids,
        prompt_attention_mask,
        end_prompt_ids,
        end_prompt_attention_mask,
        speech_quality_ids,
        speech_quality_attention_mask,
    ):
        """
        Returns query_embeds + query_attention_mask for generation.
        """
        noisy_tokens = self.audio_encoder(noisy_features).last_hidden_state
        projected_noisy_audio = self.audio_projection_layer(noisy_tokens)

        reference_tokens = self.audio_encoder(reference_features).last_hidden_state
        projected_reference_audio = self.audio_projection_layer(reference_tokens)

        tok_emb = self.llm.get_input_embeddings()
        speech_quality_embeds = tok_emb(speech_quality_ids)
        prompt_embeds = tok_emb(prompt_ids)
        end_prompt_embeds = tok_emb(end_prompt_ids)

        bs, noisy_audio_token_len = projected_noisy_audio.shape[:2]
        _, reference_audio_token_len = projected_reference_audio.shape[:2]

        query_embeds = torch.cat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                projected_reference_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
            ),
            dim=1,
        )
        query_attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                torch.ones(bs, reference_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return query_embeds, query_attention_mask


def load_modules():
    """
    Mirror llama_training2: load 4-bit LLAMA + Whisper encoder.
    """
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

    whisper_encoder = WhisperModel.from_pretrained("openai/whisper-large-v2", cache_dir="./whisper-large").get_encoder()

    return llm, whisper_encoder


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model weights from a Trainer checkpoint.
    """
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

    #print("  Missing keys:", missing)
    #print("  Unexpected keys:", unexpected)


def pad_sequence_start(sequences, batch_first=False, padding_value=0.0):
    """
    Left-align content and pad at the end to max length.
    """
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


def collate_fn_eval(batch):
    """
    Collate function for evaluation (mirrors training collate_fn).
    """
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=WHISPER_DIR)
    noisy_feature = [whisper_processor.feature_extractor.pad({"input_features": item['noisy_features']}, return_tensors="pt").input_features for item in batch]
    reference_feature = [whisper_processor.feature_extractor.pad({"input_features": item['reference_features']}, return_tensors="pt").input_features for item in batch]

    noisy_features = torch.stack(noisy_feature)
    reference_features = torch.stack(reference_feature)

    speech_quality_ids = [item["speech_quality_ids"] for item in batch]
    speech_quality_attention_mask = [
        item["speech_quality_attention_mask"] for item in batch
    ]
    prompt_ids = [item["prompt_ids"] for item in batch]
    prompt_attention_mask = [item["prompt_attention_mask"] for item in batch]
    end_prompt_ids = [item["end_prompt_ids"] for item in batch]
    end_prompt_attention_mask = [
        item["end_prompt_attention_mask"] for item in batch
    ]

    speech_quality_ids = pad_sequence_start(
        [torch.tensor(x) for x in speech_quality_ids],
        batch_first=True,
        padding_value=TOKENIZER.pad_token_id,
    )
    speech_quality_attention_mask = pad_sequence_start(
        [torch.tensor(x) for x in speech_quality_attention_mask],
        batch_first=True,
        padding_value=0,
    )
    prompt_ids = pad_sequence_start(
        [torch.tensor(x) for x in prompt_ids],
        batch_first=True,
        padding_value=TOKENIZER.pad_token_id,
    )
    prompt_attention_mask = pad_sequence_start(
        [torch.tensor(x) for x in prompt_attention_mask],
        batch_first=True,
        padding_value=0,
    )
    end_prompt_ids = pad_sequence_start(
        [torch.tensor(x) for x in end_prompt_ids],
        batch_first=True,
        padding_value=TOKENIZER.pad_token_id,
    )
    end_prompt_attention_mask = pad_sequence_start(
        [torch.tensor(x) for x in end_prompt_attention_mask],
        batch_first=True,
        padding_value=0,
    )

    labels = [item["labels"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    return {
        "noisy_features": noisy_features,
        "reference_features": reference_features,
        "speech_quality_ids": speech_quality_ids,
        "speech_quality_attention_mask": speech_quality_attention_mask,
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "end_prompt_ids": end_prompt_ids,
        "end_prompt_attention_mask": end_prompt_attention_mask,
        "labels": labels,
        "questions": questions,
        "answers": answers,
    }


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

_float_regex = re.compile(r"[-+]?\d*\.?\d+")
DIM_ORDER_MULTI = ["mos", "noi", "col", "dis", "loud"]


def extract_floats(text: str, min_val=0.0, max_val=6.0):
    """
    Extract plausible MOS/dimension values from the entire generated string.
    """
    nums = []
    for m in _float_regex.findall(text):
        try:
            x = float(m)
            if min_val <= x <= max_val:
                nums.append(x)
        except ValueError:
            continue
    return nums


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "spearman_rho": None,
        }

    assert y_true.shape == y_pred.shape
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if y_true.size >= 2:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        rho, _ = spearmanr(y_true, y_pred)
        spearman_rho = float(rho)
    else:
        pearson_r = None
        spearman_rho = None

    return {
        "n": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


class MetricBucket:
    """
    Track losses + failures for each (task, dim).
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.n_total = 0
        self.n_no_number = 0

    def add_success(self, y_true, y_pred):
        self.n_total += 1
        self.y_true.append(float(y_true))
        self.y_pred.append(float(y_pred))

    def add_failure_no_number(self):
        self.n_total += 1
        self.n_no_number += 1

    def summary(self):
        base = compute_regression_metrics(self.y_true, self.y_pred)
        base["n_total"] = int(self.n_total)
        base["n_no_number"] = int(self.n_no_number)
        return base


def evaluate_single_task(
    task_name: str,
    model: SpeechQualityLLM,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
):
    """
    Evaluate the model on a dataset where allowed_tasks=(task_name,).
    Returns:
        metrics_task: dict[dim -> metrics]
        sample_stats_multi_dim: dict (only meaningful for multi_dim)
        prediction_rows: list of per-sample dicts (for CSV)
    """
    model.eval()

    buckets = defaultdict(MetricBucket)
    multi_dim_sample_stats = {"n_samples": 0, "n_full": 0, "n_partial": 0, "n_none": 0}
    prediction_rows = []

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(test_loader, desc=f"Task={task_name}")):
            if max_batches is not None and bidx >= max_batches:
                break

            labels_list = batch["labels"]
            questions = batch["questions"]
            gt_answers = batch["answers"]

            # Prepare model inputs
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k
                in {
                    "noisy_features",
                    "reference_features",
                    "prompt_ids",
                    "prompt_attention_mask",
                    "end_prompt_ids",
                    "end_prompt_attention_mask",
                    "speech_quality_ids",
                    "speech_quality_attention_mask",
                }
            }
            model_inputs = move_to_device(model_inputs, device)

            # Encode and generate
            query_embeds, query_attention_mask = model.forward_without_llm(
                **model_inputs
            )

            gen_ids = model.llm.generate(
                inputs_embeds=query_embeds,
                attention_mask=query_attention_mask,
                max_length=query_embeds.shape[1] + 64,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )
            generated = TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)

            for i, (lbl, q, gt_ans, gen_text) in enumerate(
                zip(labels_list, questions, gt_answers, generated)
            ):
                qa_task = lbl.get("qa_task", "")
                qa_dim = lbl.get("qa_dim", None)

                floats = extract_floats(gen_text, min_val=0.0, max_val=6.0)

                row_common = {
                    "task": qa_task,
                    "dim_from_meta": qa_dim if qa_dim is not None else "",
                    "question": q,
                    "gt_answer_template": gt_ans,
                    "generated": gen_text,
                }

                if task_name == "mos_numeric":
                    dim = "mos"
                    bucket = buckets[dim]
                    y_true = lbl["mos"]

                    if len(floats) >= 1:
                        y_pred = floats[0]
                        bucket.add_success(y_true, y_pred)
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "status": "success",
                        }
                    else:
                        bucket.add_failure_no_number()
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": None,
                            "status": "no_number",
                        }
                    prediction_rows.append(row)

                elif task_name == "dim_numeric":
                    dim = qa_dim
                    if dim not in ["noi", "col", "dis", "loud"]:
                        continue
                    bucket = buckets[dim]
                    y_true = lbl[dim]

                    if len(floats) >= 1:
                        y_pred = floats[0]
                        bucket.add_success(y_true, y_pred)
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "status": "success",
                        }
                    else:
                        bucket.add_failure_no_number()
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": None,
                            "status": "no_number",
                        }
                    prediction_rows.append(row)

                elif task_name == "dim_categ":
                    # We still mine numeric info if present (e.g., "≈3.5/5").
                    dim = qa_dim
                    if dim not in ["noi", "col", "dis", "loud"]:
                        continue
                    bucket = buckets[dim]
                    y_true = lbl[dim]

                    if len(floats) >= 1:
                        y_pred = floats[0]
                        bucket.add_success(y_true, y_pred)
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "status": "success",
                        }
                    else:
                        bucket.add_failure_no_number()
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": None,
                            "status": "no_number",
                        }
                    prediction_rows.append(row)

                elif task_name == "multi_dim":
                    multi_dim_sample_stats["n_samples"] += 1
                    if len(floats) > 5: # Check for other floats
                        floats = floats[-5:]
                    n_found = len(floats)
                    if n_found == 0:
                        multi_dim_sample_stats["n_none"] += 1
                    elif n_found < len(DIM_ORDER_MULTI):
                        multi_dim_sample_stats["n_partial"] += 1
                    else:
                        multi_dim_sample_stats["n_full"] += 1

                    # For each dimension in fixed order, either use a float (if present)
                    # or mark as "no_number_in_dim".
                    for j, dim in enumerate(DIM_ORDER_MULTI):
                        bucket = buckets[dim]
                        y_true = lbl[dim]
                        bucket.n_total += 1

                        if j < n_found:
                            y_pred = floats[j]
                            bucket.y_true.append(float(y_true))
                            bucket.y_pred.append(float(y_pred))
                            row = {
                                **row_common,
                                "dim": dim,
                                "y_true": y_true,
                                "y_pred": y_pred,
                                "status": "success"
                                if n_found == len(DIM_ORDER_MULTI)
                                else "partial",
                            }
                        else:
                            bucket.n_no_number += 1
                            row = {
                                **row_common,
                                "dim": dim,
                                "y_true": y_true,
                                "y_pred": None,
                                "status": "no_number_in_dim",
                            }
                        prediction_rows.append(row)

                elif task_name == "explanatory":
                    # Treat first numeric as MOS proxy.
                    dim = "mos"
                    bucket = buckets[dim]
                    y_true = lbl["mos"]

                    if len(floats) >= 1:
                        y_pred = floats[0]
                        bucket.add_success(y_true, y_pred)
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "status": "success",
                        }
                    else:
                        bucket.add_failure_no_number()
                        row = {
                            **row_common,
                            "dim": dim,
                            "y_true": y_true,
                            "y_pred": None,
                            "status": "no_number",
                        }
                    prediction_rows.append(row)

                else:
                    # Shouldn't happen because dataset was built with allowed_tasks=(task_name,)
                    row = {
                        **row_common,
                        "dim": "",
                        "y_true": None,
                        "y_pred": None,
                        "status": "unsupported_task",
                    }
                    prediction_rows.append(row)

    # Aggregate metrics
    metrics_task = {}
    for dim, bucket in buckets.items():
        metrics_task[dim] = bucket.summary()

    return metrics_task, multi_dim_sample_stats, prediction_rows


def main():
    parser = argparse.ArgumentParser(description="Per-task evaluation of SpeechQualityLLM on NISQA.")
    parser.add_argument("--csv_path", type=str, default="Dataset/NISQA_Corpus/NISQA_corpus_file.csv")
    parser.add_argument("--dataset_split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--checkpoint_path", type=str, default="results/Reference_FrozenWhisper/checkpoint-6144")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--target_duration", type=float, default=10.0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default="mos_numeric,dim_numeric,dim_categ,multi_dim,explanatory",
        help="Comma-separated list of tasks to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation/Reference_FrozenWhisper")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base modules + wrap in SpeechQualityLLM
    llm_base, whisper_encoder = load_modules()
    model = SpeechQualityLLM(whisper_encoder=whisper_encoder, llm=llm_base, pooling_length=128)
    model = model.to(device)
    load_checkpoint(model, args.checkpoint_path, device)
    model.llm.config.pad_token_id = TOKENIZER.pad_token_id

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    metrics_all = {}

    for task_name in task_list:
        print(f"\n=== Evaluating task: {task_name} ===")

        # Dataset restricted to a single task type
        test_dataset = NISQAAudioQADataset(
            csv_path=args.csv_path,
            dataset_split=args.dataset_split,
            rng_seed=1234,
            allowed_tasks=(task_name,),
            target_sr=args.target_sr,
            target_duration=args.target_duration,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_eval,
        )

        metrics_task, multi_stats, prediction_rows = evaluate_single_task(
            task_name, model, test_loader, device, max_batches=args.max_batches
        )

        # Save per-task artifacts
        metrics_path = os.path.join(args.output_dir, f"metrics_{task_name}.json")
        out_obj = {
            "task": task_name,
            "dim_metrics": metrics_task,
            "multi_dim_sample_stats": multi_stats,
        }
        with open(metrics_path, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Saved metrics for {task_name} to {metrics_path}")

        preds_path = os.path.join(args.output_dir, f"predictions_{task_name}.csv")
        df_preds = pd.DataFrame(prediction_rows)
        df_preds.to_csv(preds_path, index=False)
        print(f"Saved per-sample predictions for {task_name} to {preds_path}")

        metrics_all[task_name] = out_obj

        print(f"\nSummary for task={task_name}:")
        for dim, m in metrics_task.items():
            mae = m["mae"]
            rmse = m["rmse"]
            pr = m["pearson_r"]
            sr = m["spearman_rho"]

            mae_str = f"{mae:.3f}" if mae is not None else "nan"
            rmse_str = f"{rmse:.3f}" if rmse is not None else "nan"
            pr_str = f"{pr:.3f}" if pr is not None else "nan"
            sr_str = f"{sr:.3f}" if sr is not None else "nan"

            print(
                f"  dim={dim:5s} | "
                f"n={m['n']}/{m['n_total']} (no_number={m['n_no_number']}) | "
                f"MAE={mae_str} RMSE={rmse_str} "
                f"Pearson={pr_str} Spearman={sr_str}"
            )


    
    metrics_all_path = os.path.join(args.output_dir, "metrics_all.json")
    with open(metrics_all_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nSaved global metrics summary to {metrics_all_path}")


if __name__ == "__main__":
    main()
