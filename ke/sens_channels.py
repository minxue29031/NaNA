import os
import torch
import json
import matplotlib.pyplot as plt
from scripts.run_circuit import extract_circuit
from block_interp.interp_mlp import load_model_and_embeddings



def plot_vote_differences(vote_old, vote_new, layer_idx, out_dir, top_n=30):
    """
    Plot channel contribution differences for a single layer as a horizontal bar chart.
    Red bars indicate positive difference, blue bars indicate negative difference.
    """
    os.makedirs(out_dir, exist_ok=True)
    diff = vote_new - vote_old

    # Select top-N channels by absolute difference
    top_vals, top_idx = torch.topk(diff.abs(), min(top_n, len(diff)))
    top_idx = top_idx.tolist()
    top_vals = diff[top_idx].tolist()


    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(top_idx)),
        top_vals,
        color=["#1f77b4" if v < 0 else "#d62728" for v in top_vals]
    )
    plt.xticks(range(len(top_idx)), [f"Ch {i}" for i in top_idx], rotation=45)
    plt.ylabel("Δ Contribution (new - old)")
    plt.xlabel("Channel")
    plt.title(f"Layer {layer_idx}: Top-{top_n} Sensitive Channels")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(out_dir, f"layer_{layer_idx}_vote_diff.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Saved plot: {save_path}")


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

    votes_old_dict = {
        item["subspace_index"] - 1: item["contribution"]
        for item in cirpoint_ori[layer_key]["subspace_results"]
    }
    votes_new_dict = {
        item["subspace_index"] - 1: item["contribution"]
        for item in cirpoint_new[layer_key]["subspace_results"]
    }

    subspaces_old = list(votes_old_dict.keys())
    subspaces_new = list(votes_new_dict.keys())

    # Compute intersection & union of channels
    inter_channels = sorted(list(set(subspaces_old) & set(subspaces_new)))
    union_channels = sorted(list(set(subspaces_old) | set(subspaces_new)))

    # Align votes for intersection 
    votes_old_aligned_inter = torch.tensor(
        [votes_old_dict[i] for i in inter_channels if i in votes_old_dict],
        dtype=torch.float32,
    )
    votes_new_aligned_inter = torch.tensor(
        [votes_new_dict[i] for i in inter_channels if i in votes_new_dict],
        dtype=torch.float32,
    )

    # Align votes for union (use 0.0 for missing channels)
    votes_old_aligned_union = torch.tensor(
        [votes_old_dict.get(i, 0.0) for i in union_channels],
        dtype=torch.float32,
    )
    votes_new_aligned_union = torch.tensor(
        [votes_new_dict.get(i, 0.0) for i in union_channels],
        dtype=torch.float32,
    )

    # Plot top channel differences
    plot_vote_differences(votes_old_aligned_union, votes_new_aligned_union, layer, out_dir=output_dir)

    # Save aligned votes and channels as JSON
    result = {
        "layer": layer,
        "subspaces_old": subspaces_old,
        "subspaces_new": subspaces_new,
        "intersection_channels": inter_channels,
        "union_channels": union_channels,
        "votes_old_aligned_inter": votes_old_aligned_inter.tolist(),
        "votes_new_aligned_inter": votes_new_aligned_inter.tolist(),
        "votes_old_aligned_union": votes_old_aligned_union.tolist(),
        "votes_new_aligned_union": votes_new_aligned_union.tolist(),
    }

    save_path = os.path.join(output_dir, f"channel_votes_layer_{layer}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n  Layer {layer} processed. Results saved to: {save_path}")

    return result
