from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
import argparse
import os


def train_tokenizer(data_file, vocab_size=16384, output_dir="tokenizer"):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    )

    tokenizer.train([data_file], trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ]
    )

    output_path = f"{output_dir}/tokenizer.json"
    tokenizer.save(output_path)
    print(f"[DONE] Tokenizer saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to tags.txt file")
    parser.add_argument("--vocab_size", type=int, default=16384, help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default="tokenizer", help="Output directory")
    args = parser.parse_args()

    train_tokenizer(args.data_file, args.vocab_size, args.output_dir)

