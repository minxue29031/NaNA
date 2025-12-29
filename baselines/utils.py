import json
import torch
import os
from transformers import GPT2Tokenizer
from baselines import config

def load_dataset(data_path, model):
    """Loads the prompt dataset and tokenizes targets."""
    print("Loading Dataset...")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    prompts = []
    target_tokens = []
    # Sort keys to ensure deterministic order
    for k in sorted(dataset.keys(), key=lambda x: int(x)):
        prompts.append(dataset[k]["prompt"])
        target_tokens.append(dataset[k]["target_token"])

    # Convert target tokens to IDs
    target_ids = torch.tensor([model.to_single_token(t) for t in target_tokens]).to(config.DEVICE)
    return prompts, target_ids

def load_baseline_json(prob_path, rank_path):
    """Loads external baseline results from JSON files."""
    if not os.path.exists(prob_path) or not os.path.exists(rank_path):
        print(f"Warning: Baseline files not found:\n {prob_path}\n {rank_path}")
        return [], [], []

    with open(prob_path, 'r') as f:
        prob_data = json.load(f)
    with open(rank_path, 'r') as f:
        rank_data = json.load(f)
        
    keys = sorted([int(k) for k in prob_data.keys() if k in rank_data], key=lambda x: x)
    probs = [prob_data[str(k)] for k in keys]
    ranks = [rank_data[str(k)] for k in keys]
    return keys, probs, ranks

def get_metrics(logits, targets):
    """Calculates average probability and rank of target tokens."""
    logits = logits[:, -1, :] 
    probs = torch.softmax(logits, dim=-1)
    
    target_probs = probs[torch.arange(len(probs)), targets]
    sorted_indices = torch.argsort(probs, dim=-1, descending=True)
    
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1]
    ranks = ranks.float() + 1  # 1-based indexing
    
    return target_probs.mean().item(), ranks.mean().item()