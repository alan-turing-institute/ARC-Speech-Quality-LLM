import argparse
import os

from src.training_noref import train as NoReferenceWithASTTraining
from src.training_ref import train as FullReferenceWithASTTraining
from src.training_ref_whisper import train as FullReferenceWithWhisperTraining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training of SpeechQualityLLM on NISQA."
    )
    parser.add_argument(
        "--training_type",
        type=str,
        default="full_reference_ast",
        choices=["full_reference_ast", "no_reference_ast", "full_reference_whisper"],
    )
    parser.add_argument(
        "--csv_path", type=str, default="Dataset/NISQA_Corpus/NISQA_corpus_file.csv"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--freeze_encoder", type=bool, default=True)
    parser.add_argument("--audio_token_length", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=2048)
    parser.add_argument("--test_steps", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument(
        "--output_dir", type=str, default="./results/FullReference_FrozenAST_temp"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="Speech-Quality-Expert-LLM-Reference-FrozenAST-Temp",
    )
    args = parser.parse_args()

    print("Running training ----------")

    print(f"Training type: {args.training_type}")

    if args.training_type == "full_reference_ast":
        FullReferenceWithASTTraining(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            freeze_encoder=args.freeze_encoder,
            audio_token_length=args.audio_token_length,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            test_steps=args.test_steps,
            project_name=args.project_name,
        )
    elif args.training_type == "no_reference_ast":
        NoReferenceWithASTTraining(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            freeze_encoder=args.freeze_encoder,
            audio_token_length=args.audio_token_length,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            test_steps=args.test_steps,
            project_name=args.project_name,
        )
    elif args.training_type == "full_reference_whisper":
        FullReferenceWithWhisperTraining(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            freeze_encoder=args.freeze_encoder,
            audio_token_length=args.audio_token_length,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            test_steps=args.test_steps,
            project_name=args.project_name,
        )
    else:
        raise ValueError("Training mode not yet available!")
