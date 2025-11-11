import os
import torch
from block_interp.model_load import get_mlp_matrices
from block_interp.mlp_svd_utils import compute_svd
from ke.sens_channels import select_sensitive_channels
from block_interp.interp_mlp import load_model_and_embeddings



def edit_single_mlp_layer(
    W,
    layer_idx,
    subspaces_old,
    subspaces_new,
    votes_old_union,
    votes_new_union,
    weight_type,
    delta_boost=1.2,
    delta_suppress=0.8,
):
    """
    Edit a single MLP layer based on differences between new/old unique subspace contributions:
      - New-only channels: boost or suppress depending on contribution sign
      - Old-only channels: uniformly suppress
      - Shared channels: leave unchanged
    """
    
    # Get the MLP weight matrices
    c_fc, c_proj, ln_2, act = get_mlp_matrices(W, layer_idx)
    U, S, Vh = compute_svd(weight_type, c_fc=c_fc, c_proj=c_proj, ln_2=ln_2)

    # Extract original weights depending on weight type
    if weight_type == "c_proj":
        W_orig = c_proj.weight.detach()
    else:
        W_orig = (c_fc.weight.detach().T * ln_2.weight.detach()).T

    W_edited = W_orig.clone()

    # Convert subspace lists to sets for set operations
    subspaces_old = set(subspaces_old)
    subspaces_new = set(subspaces_new)
    inter_channels = subspaces_old & subspaces_new
    new_only_channels = sorted(list(subspaces_new - inter_channels))
    old_only_channels = sorted(list(subspaces_old - inter_channels))

    print(f"\n=== Editing layer {layer_idx} ===")
    print(f"  → new-only: {len(new_only_channels)}, old-only: {len(old_only_channels)}, shared: {len(inter_channels)}")

    # Adjust new-only channels based on contribution sign
    for idx, ch in enumerate(new_only_channels):
        if ch >= len(S):
            continue
        contrib = votes_new_union[idx].item() if idx < len(votes_new_union) else 0.0

        if contrib > 0:
            s_value = S[ch] * (delta_boost - 1)
            action = "BOOST"
        elif contrib < 0:
            s_value = -S[ch] * (1 - delta_suppress)
            action = "SUPPRESS"
        else:
            continue

        print(f"  >> [NEW] Channel {ch:4d} | contrib={contrib:+.4f} | action={action} | Δ={s_value:+.4f}")
        if weight_type == "c_proj":
            W_edited += s_value * torch.outer(U[:, ch], Vh[ch, :])
        else:
            W_edited += (s_value * torch.outer(U[:, ch], Vh[ch, :])).T

    # Uniformly suppress old-only channels
    for idx, ch in enumerate(old_only_channels):
        if ch >= len(S):
            continue
        contrib = votes_new_union[idx].item() if idx < len(votes_new_union) else 0.0

        if contrib > 0:
            s_value = -S[ch] * (1 - delta_suppress)
            action = "SUPPRESS"
        else:
            continue

        print(f"  >> [OLD] Channel {ch:4d} | contrib={contrib:+.4f} | action={action} | Δ={s_value:+.4f}")
        if weight_type == "c_proj":
            W_edited += s_value * torch.outer(U[:, ch], Vh[ch, :])
        else:
            W_edited += (s_value * torch.outer(U[:, ch], Vh[ch, :])).T

    print(f" ===== Finished editing layer {layer_idx} =====")
    return W_edited


def edit_mlp_layers(
    W,
    model_name,
    in_seq,
    ori_target,
    new_target,
    layers,
    weight_type="c_fc",
    delta_boost=1.2,
    delta_suppress=0.8,
    device="cuda",
    interp_type="all",
    circuit_mode="DeEf",
    topk_subspaces=15,
    output_dir="result/ke"
):
    """
    Edit multiple MLP layers using boosts/suppression based on unique new/old channels.
    """
    os.makedirs(output_dir, exist_ok=True)
    edited_weights = {}

    for layer_idx in layers:
        print(f"\n [Layer {layer_idx}] Computing new/old intersection ===")

        # Get sensitive channels and contributions
        result = select_sensitive_channels(
            model_name=model_name,
            in_seq=in_seq,
            ori_target=ori_target,
            new_target=new_target,
            layer=layer_idx,
            topk_subspaces=topk_subspaces,
            device=device,
            weight_type=weight_type,
            interp_type=interp_type,
            circuit_mode=circuit_mode,
            output_dir=output_dir,
        )

        subspaces_old = result["subspaces_old"]
        subspaces_new = result["subspaces_new"]
        votes_old_union = torch.tensor(result["votes_old_aligned_union"])
        votes_new_union = torch.tensor(result["votes_new_aligned_union"])

        print(f"  -> subspaces_old: {len(subspaces_old)}, subspaces_new: {len(subspaces_new)}")

        # Edit the single layer
        W_edited = edit_single_mlp_layer(
            W,
            layer_idx,
            subspaces_old,
            subspaces_new,
            votes_old_union,
            votes_new_union,
            weight_type,
            delta_boost=delta_boost,
            delta_suppress=delta_suppress,
        )

        edited_weights[layer_idx] = W_edited
        print(f" [Layer {layer_idx}] Edited successfully.\n")

    return edited_weights
