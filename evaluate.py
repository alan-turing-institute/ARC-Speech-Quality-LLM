import os
import argparse

import torch
import torch.nn as nn
from safetensors.torch import load_file

from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.dataloader_ref import (
    AST_DIR,
    LLAMA_DIR,
    TOKENIZER,
    AST_FEATURE_EXTRACTOR,
    load_wav_mono,
    align_and_window_pair,
    prompt_template_fn,
    end_template,
)


if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token


class AudioProjectionLayer(nn.Module):
    """
    Same audio projection layer as in evaluate_ref.py:
    - Pool along time
    - LayerNorm over channel dim
    - Linear projection into LLaMA hidden size
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
        # x: [B, T, C_in]
        x = x.transpose(-2, -1)          # [B, C_in, T]
        x = self.pool(x)                 # [B, C_in, T_p]
        x = x.transpose(-2, -1)          # [B, T_p, C_in]
        x = self.layer_norm(x)
        x = self.linear_projection(x)    # [B, T_p, C_out]
        return x


class SpeechQualityLLM(nn.Module):

    def __init__(self, ast_encoder, llm, pooling_length=128):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.audio_encoder = ast_encoder

        # LoRA configuration must match training (r, alpha, target_modules, etc.).
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
            input_embedding_size=768,
            output_embedding_size=4096,
            temporal_pooling=pooling_length,
        )

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
        Build the query embeddings from:
          prompt + projected(noisy_audio) + projected(reference_audio) + end_prompt.
        """
        # AST encoder: input [B, 1, T] -> last_hidden_state [B, T', C]
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

        # [B, L_prompt + L_noisy + L_ref + L_end, D]
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
                torch.ones(bs, noisy_audio_token_len, device=speech_quality_ids.device),
                torch.ones(bs, reference_audio_token_len, device=speech_quality_ids.device),
                end_prompt_attention_mask,
            ),
            dim=1,
        )

        return query_embeds, query_attention_mask


def load_modules():
    """
    Same as in evaluate_ref.py: load 4-bit LLaMA + AST encoder.
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

    ast_encoder = AutoModelForAudioClassification.from_pretrained(
        AST_DIR
    ).audio_spectrogram_transformer

    return llm, ast_encoder


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    """
    Load model weights from Trainer checkpoint directory or .safetensors file.
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
    ref_path: str,
    prompt: str,
    system_message: str,
    target_sr: int,
    target_duration: float,
):
    """
    Build a single-example 'batch' dict with the same keys as collate_fn_eval()
    expects in evaluate_ref.py, but using manual file paths + prompt instead of NISQA CSV.

    Returns a dict with:
      noisy_features, reference_features,
      speech_quality_ids, speech_quality_attention_mask,
      prompt_ids, prompt_attention_mask,
      end_prompt_ids, end_prompt_attention_mask
    """
    # 1) Load mono audio and resample
    wav_deg, sr_deg = load_wav_mono(deg_path, target_sr=target_sr)
    wav_ref, sr_ref = load_wav_mono(ref_path, target_sr=target_sr)

    if sr_deg != target_sr or sr_ref != target_sr:
        raise ValueError(
            f"Resampling failed? sr_deg={sr_deg}, sr_ref={sr_ref}, target_sr={target_sr}"
        )

    # 2) Time-align and window both signals
    wav_ref_win, wav_deg_win = align_and_window_pair(
        wav_ref,
        wav_deg,
        sr=target_sr,
        target_duration=target_duration,
        rng=None,
        max_lag_seconds=1.0,
    )

    # 3) Extract AST input features (degraded + reference)
    feature_deg = AST_FEATURE_EXTRACTOR(
        wav_deg_win,
        sampling_rate=target_sr
    )["input_values"][0]
    feature_ref = AST_FEATURE_EXTRACTOR(
        wav_ref_win,
        sampling_rate=target_sr
    )["input_values"][0]

    noisy_feature = AST_FEATURE_EXTRACTOR.pad(
        {"input_values": feature_deg},
        return_tensors="pt",
    ).input_values
    reference_feature = AST_FEATURE_EXTRACTOR.pad(
        {"input_values": feature_ref},
        return_tensors="pt",
    ).input_values

    noisy_features = noisy_feature.unsqueeze(0)
    reference_features = reference_feature.unsqueeze(0)

    # 4) Text: system + user prompt, end-of-prompt marker, and dummy answer tokens
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
        "noisy_features": noisy_features,                 # [1, 1, T]
        "reference_features": reference_features,         # [1, 1, T]
        "speech_quality_ids": speech_quality_ids,         # [1, L_ans]
        "speech_quality_attention_mask": speech_quality_attention_mask,
        "prompt_ids": prompt_ids,                         # [1, L_prompt]
        "prompt_attention_mask": prompt_attention_mask,
        "end_prompt_ids": end_prompt_ids,                 # [1, L_end]
        "end_prompt_attention_mask": end_prompt_attention_mask,
    }
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Interactive audio-quality QA with SpeechQualityLLM"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/Reference/checkpoint-10240",
        help="Path to Trainer checkpoint directory or .safetensors file.",
    )
    parser.add_argument(
        "--deg_path",
        type=str,
        required=True,
        help="Path to degraded audio file.",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        required=True,
        help="Path to reference audio file.",
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
        help="Target window length (seconds) after alignment.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.deg_path):
        raise FileNotFoundError(f"Degraded audio not found: {args.deg_path}")
    if not os.path.isfile(args.ref_path):
        raise FileNotFoundError(f"Reference audio not found: {args.ref_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base models and wrap in SpeechQualityLLM
    llm_base, ast_encoder = load_modules()
    model = SpeechQualityLLM(ast_encoder=ast_encoder, llm=llm_base, pooling_length=128)
    model = model.to(device)
    load_checkpoint(model, args.checkpoint_path)
    model.llm.config.pad_token_id = TOKENIZER.pad_token_id

    # Build single-example batch from file paths + prompt
    batch = build_single_example_batch(
        deg_path=args.deg_path,
        ref_path=args.ref_path,
        prompt=args.prompt,
        system_message=args.system_message,
        target_sr=args.target_sr,
        target_duration=args.target_duration,
    )
    batch = move_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        # Encode audio + prompt into query embeddings
        query_embeds, query_attention_mask = model.forward_without_llm(
            noisy_features=batch["noisy_features"],
            reference_features=batch["reference_features"],
            prompt_ids=batch["prompt_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            end_prompt_ids=batch["end_prompt_ids"],
            end_prompt_attention_mask=batch["end_prompt_attention_mask"],
            speech_quality_ids=batch["speech_quality_ids"],
            speech_quality_attention_mask=batch["speech_quality_attention_mask"],
        )

        # Generate answer tokens conditioned on (audio, prompt)
        gen_ids = model.llm.generate(
            inputs_embeds=query_embeds,
            attention_mask=query_attention_mask,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
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
