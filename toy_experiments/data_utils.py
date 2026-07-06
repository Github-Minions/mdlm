"""
Data Preparation Utilities
============================
Functions to download, cache, and prepare TinyStories dataset
for MDLM training and evaluation.
"""

import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer


def download_tinystories(cache_dir="./cache"):
    """
    Download TinyStories dataset from HuggingFace.
    Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
    """
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    print(f"Train size: {len(dataset['train'])}")
    print(f"Valid size: {len(dataset['validation'])}")
    return dataset


def prepare_validation_set(dataset, n_samples=100, seed=1, output_path=None):
    """
    Extract a fixed validation set for downstream evaluation.
    This ensures reproducibility across all experiments.

    Args:
        dataset: HuggingFace dataset
        n_samples: Number of validation texts (default 100, matching the report)
        seed: Random seed for reproducibility
        output_path: Where to save the validation set

    Returns:
        List of text strings
    """
    torch.manual_seed(seed)
    val_data = dataset['validation']
    indices = torch.randperm(len(val_data))[:n_samples].tolist()
    texts = [val_data[i]['text'] for i in indices]

    if output_path:
        torch.save(texts, output_path)
        print(f"Validation set saved to {output_path}")

    return texts


def prepare_tokenizer(model_name='gpt2'):
    """
    Prepare GPT-2 tokenizer for TinyStories.
    GPT-2 vocab size: 50257
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.mask_token is None:
        # Add mask token for MDLM training
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer


def get_pretrained_checkpoints():
    """
    Download pre-trained checkpoints from goosemaths HuggingFace repository.
    Alternative to training from scratch.

    URL: https://huggingface.co/datasets/goosemaths/tinystories_masked_diffusion_model_ckpt
    """
    from huggingface_hub import hf_hub_download

    repo_id = "goosemaths/tinystories_masked_diffusion_model_ckpt"
    cache_dir = "./checkpoints"
    os.makedirs(cache_dir, exist_ok=True)

    checkpoints = {
        'mdlm_step100': 'mdlm_step100.ckpt',
        'sedd': 'sedd.ckpt',
    }

    downloaded = {}
    for name, filename in checkpoints.items():
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   local_dir=cache_dir)
            downloaded[name] = path
            print(f"Downloaded {name}: {path}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    return downloaded


if __name__ == '__main__':
    print("="*60)
    print("TinyStories Data Preparation")
    print("="*60)

    # Step 1: Download dataset
    dataset = download_tinystories()

    # Step 2: Prepare validation set
    val_texts = prepare_validation_set(
        dataset, n_samples=100, seed=1,
        output_path='./cache/val_set_100.pt')

    # Step 3: Prepare tokenizer
    tokenizer = prepare_tokenizer()

    # Step 4: (Optional) Download pre-trained checkpoints
    print("\nTo download pre-trained checkpoints, run:")
    print("  python data_utils.py --download-checkpoints")

    import sys
    if '--download-checkpoints' in sys.argv:
        get_pretrained_checkpoints()
