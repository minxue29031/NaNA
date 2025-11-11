import os
import torch
import json
import matplotlib.pyplot as plt
from scripts.run_circuit import extract_circuit
from block_interp.interp_mlp import load_model_and_embeddings

 

def select_sensitive_channels(
    model_name="gpt2-medium",
    in_seq="The cat looks very",
    ori_target=" happy",
    new_target=" cute",
    layer=16,
    topk_subspaces=15,
    device="cuda",
    weight_type="c_fc",
    interp_type="all",
    circuit_mode="DeEf",
    output_dir="result/ke"
):
    """
    Compute single-layer subspace contribution differences, and return aligned votes
    for intersection and union of old/new channels.
    """
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)

    # Original target circuit 
    print(f"\n=== [1] Run for original target: '{ori_target}' ===")
    _, cirpoint_ori = extract_circuit(
        model,
        tokenizer,
        W_E,
        model_name=model_name,
        in_seq=in_seq,
        target_word=ori_target,
        layers=[str(layer)],
        topk_subspaces=topk_subspaces,
        output_dir=os.path.join(output_dir, "ori_target"),
        circuit_mode=circuit_mode,
        interp_type=interp_type,
        weight_type=weight_type,
        return_heatmap=False,
        do_interp=False
    )

    # New target circuit
    print(f"\n=== [2] Run for new target: '{new_target}' ===")
    _, cirpoint_new = extract_circuit(
        model,
        tokenizer,
        W_E,
        model_name=model_name,
        in_seq=in_seq,
        target_word=new_target,
        layers=[str(layer)],
        topk_subspaces=topk_subspaces,
        output_dir=os.path.join(output_dir, "new_target"),
        circuit_mode=circuit_mode,
        interp_type=interp_type,
        weight_type=weight_type,
        return_heatmap=False,
        do_interp=False
    )

    layer_key = f"layer_{layer}"
    print(f"\n=== [3] Processing {layer_key} ===")

    # Original contributions and channels
    votes_old_original = [item["contribution"] for item in cirpoint_ori[layer_key]["subspace_results"]]
    votes_new_original = [item["contribution"] for item in cirpoint_new[layer_key]["subspace_results"]]

    # New contributions and channels
    channels_old = [item["subspace_index"] - 1 for item in cirpoint_ori[layer_key]["subspace_results"]]
    channels_new = [item["subspace_index"] - 1 for item in cirpoint_new[layer_key]["subspace_results"]]
 
    result = {
        "layer": layer,
        "subspaces_old": channels_old,
        "subspaces_new": channels_new,
        "votes_old": votes_old_original,
        "votes_new": votes_new_original,
    }

    save_path = os.path.join(output_dir, f"channel_votes_layer_{layer}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n  Layer {layer} processed. Results saved to: {save_path}")

    return result