import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
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


class AudioProjectionLayer(nn.Module):
    """
    Pool along time, LayerNorm over channel dim, linear projection into LLaMA hidden size.
    """

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
        from peft import LoraConfig, get_peft_model

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
        """
        Build query embeddings from: prompt + projected(noisy_audio) + end_prompt.
        """
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
                torch.ones(bs, noisy_audio_token_len, device=speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return query_embeds, query_attention_mask


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


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_single_example_batch(
    deg_path: str,
    prompt: str,
    system_message: str,
    target_sr: int,
    target_duration: float,
):
    """
    Build a single-example batch for no-reference evaluation.
    Only the degraded audio is needed — no reference signal.
    """
    wav_deg, sr_deg = load_wav_mono(deg_path, target_sr=target_sr)

    if sr_deg != target_sr:
        raise ValueError(f"Resampling failed? sr_deg={sr_deg}, target_sr={target_sr}")

    # Window/crop to target duration
    target_len = int(target_duration * target_sr)
    if wav_deg.shape[0] >= target_len:
        wav_deg = wav_deg[:target_len]
    else:
        pad_len = target_len - wav_deg.shape[0]
        wav_deg = np.pad(wav_deg, (0, pad_len), mode="constant")

    # Extract AST input features
    feature_deg = AST_FEATURE_EXTRACTOR(wav_deg, sampling_rate=target_sr)[
        "input_values"
    ][0]

    noisy_feature = AST_FEATURE_EXTRACTOR.pad(
        {"input_values": feature_deg},
        return_tensors="pt",
    ).input_values

    noisy_features = noisy_feature.unsqueeze(0)

    # Tokenise prompt, end-of-prompt marker, and dummy answer
    full_prompt = prompt_template_fn(
        prompt=prompt,
        system_message=system_message,
    )
    prompt_tokens = TOKENIZER(
        full_prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_tokens.input_ids
    prompt_attention_mask = prompt_tokens.attention_mask

    end_prompt_tokens = TOKENIZER(
        end_template(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    end_prompt_ids = end_prompt_tokens.input_ids
    end_prompt_attention_mask = end_prompt_tokens.attention_mask

    dummy_answer = "The audio quality is: "
    sq_tokens = TOKENIZER(
        dummy_answer,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    speech_quality_ids = sq_tokens.input_ids
    speech_quality_attention_mask = sq_tokens.attention_mask

    batch = {
        "noisy_features": noisy_features,
        "speech_quality_ids": speech_quality_ids,
        "speech_quality_attention_mask": speech_quality_attention_mask,
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "end_prompt_ids": end_prompt_ids,
        "end_prompt_attention_mask": end_prompt_attention_mask,
    }
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="No-reference single-audio quality QA with SpeechQualityLLM"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/NoReference/checkpoint-10240",
        help="Path to Trainer checkpoint directory or .safetensors file.",
    )
    parser.add_argument(
        "--deg_path",
        type=str,
        required=True,
        help="Path to degraded / synthetic audio file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User question / instruction about audio quality.",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are an audio quality evaluation expert.",
        help="System role description for the LLM.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Resampling rate for AST frontend.",
    )
    parser.add_argument(
        "--target_duration",
        type=float,
        default=10.0,
        help="Target window length (seconds).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no_temperature",
        type=bool,
        default=False,
        help="Whether to disable sampling and use greedy decoding.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.deg_path):
        raise FileNotFoundError(f"Audio file not found: {args.deg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    llm_base, ast_encoder = load_modules()
    model = SpeechQualityLLM(ast_encoder=ast_encoder, llm=llm_base, pooling_length=128)
    model = model.to(device)
    load_checkpoint(model, args.checkpoint_path)
    model.llm.config.pad_token_id = TOKENIZER.pad_token_id

    batch = build_single_example_batch(
        deg_path=args.deg_path,
        prompt=args.prompt,
        system_message=args.system_message,
        target_sr=args.target_sr,
        target_duration=args.target_duration,
    )
    batch = move_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        query_embeds, query_attention_mask = model.forward_without_llm(
            noisy_features=batch["noisy_features"],
            prompt_ids=batch["prompt_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            end_prompt_ids=batch["end_prompt_ids"],
            end_prompt_attention_mask=batch["end_prompt_attention_mask"],
            speech_quality_ids=batch["speech_quality_ids"],
            speech_quality_attention_mask=batch["speech_quality_attention_mask"],
        )

        gen_ids = model.llm.generate(
            inputs_embeds=query_embeds,
            attention_mask=query_attention_mask,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            do_sample=not args.no_temperature,
        )
        generated = TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 80)
    print("PROMPT:")
    print(args.prompt)
    print("-" * 80)
    print("MODEL ANSWER:")
    print(generated)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
