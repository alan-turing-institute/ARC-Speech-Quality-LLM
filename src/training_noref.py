import gc
import os
import re
import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy.signal as sig
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    logging,
)

from src.dataloader_noref import (
    AST_DIR,
    AST_FEATURE_EXTRACTOR,
    LLAMA_DIR,
    TOKENIZER,
    NISQAAudioQADataset,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# import wandb

# wandb.login(key="X")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def load_modules():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR, quantization_config=bnb_config
    )

    ast_encoder = AutoModelForAudioClassification.from_pretrained(
        AST_DIR
    ).audio_spectrogram_transformer

    return llm, ast_encoder


"""
Debugging Options
"""


def evaluate_llm_output(
    llm,
    tokenizer,
    prompt="Tell me a dark joke.",
):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = llm.generate(inputs["input_ids"], max_length=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


"""
Model Architecture
"""


class AudioProjectionLayer(nn.Module):
    def __init__(
        self, input_embedding_size=768, output_embedding_size=4096, temporal_pooling=256
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(temporal_pooling)
        self.layer_norm = nn.LayerNorm(input_embedding_size)
        self.linear_projection = nn.Linear(
            in_features=input_embedding_size, out_features=output_embedding_size
        )

    def forward(self, x):
        """
        x: [bs, sequence length, input embedding]
        Returns:
            x: [bs, pooled sequence length, output embedding]
        """
        x = x.transpose(-2, -1)
        x = self.pool(x)

        x = x.transpose(-2, -1)
        x = self.layer_norm(x)

        x = self.linear_projection(x)

        return x


class SpeechQualityLLM(nn.Module):
    def __init__(self, ast_encoder, llm, freeze_encoder=False, pooling_length=128):
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
            if not freeze_encoder:
                if "attention.query" in name:
                    param.requires_grad = True

        self.audio_projection_layer = AudioProjectionLayer(
            input_embedding_size=768,
            output_embedding_size=4096,
            temporal_pooling=pooling_length,
        )

    def forward(
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

        input_embeds = torch.concat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
                speech_quality_embeds,
            ),
            dim=1,
        )

        query_embeds = torch.concat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
            ),
            dim=1,
        )

        attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
                speech_quality_attention_mask,
            ),
            dim=1,
        )

        query_attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        outputs = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)

        return outputs, query_embeds, query_attention_mask, noisy_audio_token_len

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

        query_embeds = torch.concat(
            (
                prompt_embeds,
                projected_noisy_audio.to(speech_quality_embeds.dtype),
                end_prompt_embeds,
            ),
            dim=1,
        )

        query_attention_mask = torch.concat(
            (
                prompt_attention_mask,
                torch.ones(bs, noisy_audio_token_len).to(speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return query_embeds, query_attention_mask


"""
Model Utility Functions
"""


def move_to_device(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def garbage_collection():
    gc.collect()
    torch.cuda.empty_cache()


def load_model_weights(save_dir, model):
    state_dict = torch.load(save_dir + "training.pth")
    model_state_dict = model.state_dict()

    for k, v in state_dict.items():
        if k in model_state_dict:
            print(k)

    updated_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(updated_state_dict)
    model.load_state_dict(model_state_dict)

    return model


"""
Dataset Utility Functions
"""


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
        padded_seq = torch.cat([seq, padding], dim=0)

        padded_seqs.append(padded_seq)

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

    speech_quality_ids = [item["speech_quality_ids"] for item in batch]
    speech_quality_attention_mask = [
        item["speech_quality_attention_mask"] for item in batch
    ]
    prompt_ids = [item["prompt_ids"] for item in batch]
    prompt_attention_mask = [item["prompt_attention_mask"] for item in batch]
    end_prompt_ids = [item["end_prompt_ids"] for item in batch]
    end_prompt_attention_mask = [item["end_prompt_attention_mask"] for item in batch]

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

    noisy_features = torch.stack([torch.tensor(x) for x in noisy_feature])

    return {
        "noisy_features": noisy_features,
        "speech_quality_ids": speech_quality_ids,
        "speech_quality_attention_mask": speech_quality_attention_mask,
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "end_prompt_ids": end_prompt_ids,
        "end_prompt_attention_mask": end_prompt_attention_mask,
    }


def verify_dataloader(dataloader):
    for batch in dataloader:
        print("Input features shape:", batch["input_features"].shape)
        print("Speech Quality IDs shape:", batch["speech_quality_ids"].shape)
        print(
            "Speech Quality attention mask shape:",
            batch["speech_quality_attention_mask"].shape,
        )
        print("Prompt IDs shape:", batch["prompt_ids"].shape)
        print("Prompt attention mask shape:", batch["prompt_attention_mask"].shape)
        print("End prompt IDs shape:", batch["end_prompt_ids"].shape)
        print(
            "End prompt attention mask shape:", batch["end_prompt_attention_mask"].shape
        )

        break


"""
Inference
"""


# def log_results_to_wandb(results):
#     table = wandb.Table(columns=["Ground Truth", "Generated"])
#     for result in results:
#         table.add_data(result["Ground Truth"], result["Generated"])
#     wandb.log({"Caption Comparison": table})


def run_inference(llm, model, dataloader, tokenizer, device, num_samples=8):
    model.eval()
    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            if num_samples is not None and i >= num_samples:
                break

            batch = move_to_device(batch, device)

            query_embeds, query_attention_mask = model.forward_without_llm(**batch)

            sampled = llm.generate(
                inputs_embeds=query_embeds,
                attention_mask=query_attention_mask,
                max_length=query_embeds.shape[1] + 64,
            )

            generated = tokenizer.batch_decode(sampled, skip_special_tokens=True)

            ground_truth = tokenizer.batch_decode(
                batch["speech_quality_ids"], skip_special_tokens=True
            )

            for gt, gen in zip(ground_truth, generated):
                results.append({"Ground Truth": gt, "Generated": gen})

    return results


class MultiModalTrainer(Trainer):
    def __init__(self, test_dataloader, tokenizer, test_steps=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataloader = test_dataloader
        self.test_step = 0
        self.tokenizer = tokenizer
        self.test_steps = test_steps

    def move_to_device(self, batch, device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and run test loop every 500 steps.
        """
        # Perform regular training step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Increment test step counter
        self.test_step += 1

        # Run test loop every 500 steps
        # for test purposes logging at every step
        if self.test_step > 0 and self.test_step % self.test_steps == 0:
            self.run_test_loop()

        return loss

    def run_test_loop(self):
        inference_results = run_inference(
            self.model.llm,
            self.model,
            self.test_dataloader,
            self.tokenizer,
            device,
            num_samples=8,
        )
        # log_results_to_wandb(inference_results)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs, _, _, audio_seq = model(**inputs)
        loss = self.calculate_loss(outputs, inputs, audio_seq)
        return (loss, outputs) if return_outputs else loss

    def calculate_loss(self, outputs, batch, audio_seq):
        logits = outputs.logits
        prompt_ids = batch["prompt_ids"]
        end_prompt_ids = batch["end_prompt_ids"]
        caption_ids = batch["speech_quality_ids"]

        prompt_ids_seq = prompt_ids.shape[1]
        end_prompt_ids_seq = end_prompt_ids.shape[1]
        logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

        op_logits = logits[:, logits_start:-1, :].contiguous()
        caption_labels = caption_ids[:, 1:].contiguous()

        if op_logits.shape[1] != caption_labels.shape[1]:
            raise ValueError(
                f"Shape mismatch: op_logits {op_logits.shape}, caption_labels {caption_labels.shape}"
            )

        loss = torch.nn.functional.cross_entropy(
            op_logits.view(-1, op_logits.shape[-1]), caption_labels.view(-1)
        )
        return loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        metrics = {"wer": 1.0}
        # Validation
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Validating"):
                batch = self.move_to_device(batch, device)
                # batch = {k: v.to(device) for k, v in batch.items()}
                outputs, _, _, audio_seq = self.model(**batch)
                loss = self.calculate_loss(outputs, batch, audio_seq)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(eval_dataloader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print()

        # Log metrics to wandb
        metrics = {"eval_loss": avg_val_loss}
        print(metrics)
        return metrics


def train(
    csv_path="Dataset/NISQA_Corpus/NISQA_corpus_file.csv",
    batch_size=4,
    freeze_encoder=False,
    audio_token_length=128,
    output_dir="./results/NoReference_FrozenAST",
    num_train_epochs=8,
    warmup_steps=128,
    eval_steps=512,
    save_steps=512,
    test_steps=512,
    project_name="Speech-Quality-Expert-LLM-NoRef-FrozenAST",
):
    llm, ast_encoder = load_modules()
    model = SpeechQualityLLM(
        ast_encoder=ast_encoder,
        llm=llm,
        freeze_encoder=freeze_encoder,
        pooling_length=audio_token_length,
    )

    pc = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            pc += param.numel()
    print(f"Total Params: {pc/1e6:.2f}M")

    model = model.to(device)

    train_dataset = NISQAAudioQADataset(
        csv_path=csv_path,
        dataset_split="TRAIN",
        rng_seed=1234,
        allowed_tasks=(
            "mos_numeric",
            "dim_numeric",
            "dim_categ",
            "multi_dim",
            "explanatory",
        ),
        target_sr=16000,
        target_duration=10.0,
    )

    val_dataset = NISQAAudioQADataset(
        csv_path=csv_path,
        dataset_split="VAL",
        rng_seed=1234,
        allowed_tasks=(
            "mos_numeric",
            "dim_numeric",
            "dim_categ",
            "multi_dim",
            "explanatory",
        ),
        target_sr=16000,
        target_duration=10.0,
    )

    test_dataset = NISQAAudioQADataset(
        csv_path=csv_path,
        dataset_split="TEST",
        rng_seed=1234,
        allowed_tasks=(
            "mos_numeric",
            "dim_numeric",
            "dim_categ",
            "multi_dim",
            "explanatory",
        ),
        target_sr=16000,
        target_duration=10.0,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # wandb.init(project=project_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=32,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        learning_rate=4e-5,
        # report_to="wandb",
    )

    # Initialize Trainer
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataloader=test_dataloader,
        tokenizer=TOKENIZER,
        test_steps=test_steps,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=False)

    # # Save the final model
    trainer.save_model()
    trainer.save_state()

    # # Finish the wandb run
    # wandb.finish()


if __name__ == "__main__":
    train()
