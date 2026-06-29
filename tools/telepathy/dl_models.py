#!/usr/bin/env python3
# dl_models.py <repo_id> — warm the HF cache for one repo (weights+config+tokenizer only).
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
pats = ["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt", "*.bin"]
p = snapshot_download(repo, allow_patterns=pats)
print("DONE", repo, "->", p)
