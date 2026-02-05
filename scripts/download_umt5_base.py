"""
Download umt5-base model from HuggingFace to local directory.
Usage: python scripts/download_umt5_base.py
"""
import os
from transformers import AutoTokenizer, AutoModel

def main():
    model_name = "google/umt5-base"
    save_dir = "deps/t5_umt5-base-enc/google/umt5-base"

    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")

    print(f"Downloading {model_name} model...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    print("Done!")

if __name__ == "__main__":
    main()
