import torch
from circuit.model_interface import project_mlp_acv_to_vocab


def subspace_interv(
    model,
    tokenizer,
    U, S, Vh,
    layer_io,
    layer_idx,
    W_E,
    reshape_W_E,
    weight_type,
    interv_mode="enhance",
    interv_dir_indices=[6],
    interv_scale=0.8,
    return_toptoks=20,
):
    """
    Apply SVD-based subspace intervention on MLP (supports c_fc and c_proj).
    """

    ori_top_tokens_list = project_mlp_acv_to_vocab(
        model,
        tokenizer,
        layer_io["output"],
        topk=return_toptoks,
        weight_type=weight_type,
        W_E=W_E,
        reshape_W_E=reshape_W_E,
    )

    if interv_mode == "remove":
        print(f"\n>> Removing subspace directions {interv_dir_indices}")
        top_tokens_list = removed_subspaces_proj(
            model,
            tokenizer,
            U,
            S,
            Vh,
            layer_io,
            layer_idx,
            weight_type=weight_type,
            W_E=W_E,
            reshape_W_E=reshape_W_E,
            remove_indices=interv_dir_indices,
            topk=return_toptoks,
        )

    elif interv_mode == "enhance":
        print(f"\n>> Enhancing subspace directions {interv_dir_indices} (scale={interv_scale})")
        top_tokens_list = enhanced_subspaces_proj(
            model,
            tokenizer,
            U,
            S,
            Vh,
            layer_io,
            layer_idx,
            weight_type=weight_type,
            W_E=W_E,
            reshape_W_E=reshape_W_E,
            enhance_indices=interv_dir_indices,
            enhance_scale=interv_scale,
            topk=return_toptoks,
        )

    else:
        raise ValueError("interv_mode must be either 'enhance' or 'remove'.")

    return top_tokens_list


def get_mlp_weight(model, layer_idx, weight_type):
    mlp = model.transformer.h[layer_idx].mlp
    if weight_type == "c_fc":
        return mlp.c_fc.weight.T.detach().cpu()
    elif weight_type == "c_proj":
        return mlp.c_proj.weight.detach().cpu()
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")


def removed_subspaces_proj(
    model,
    tokenizer,
    U,
    S,
    Vh,
    layer_io,
    layer_idx,
    weight_type="c_proj",
    W_E=None,
    reshape_W_E=None,
    remove_indices=None,
    topk=20,
):
    if remove_indices is None:
        remove_indices = []
    remove_indices = set(remove_indices)

    print(f"\nRemoving principal components {sorted(remove_indices)} from {weight_type} (layer {layer_idx})...")

    W = get_mlp_weight(model, layer_idx, weight_type)

    W_reduced = torch.zeros_like(W)
    for i in range(S.shape[0]):
        if i in remove_indices:
            continue
        W_reduced += S[i] * torch.outer(U[:, i], Vh[i, :])

    x = layer_io["input"].cpu()

    # Compute layer output depending on weight type
    y_reduced = x @ W_reduced

    # Project to vocab depending on which matrix we are working on
    if weight_type == "c_fc":
        logits_proj = y_reduced @ reshape_W_E.T
    else:
        logits_proj = y_reduced @ W_E.T

    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1)
    tokens_scores = [(tokenizer.decode([idx.item()]), score.item()) for idx, score in zip(top_idx[0], top_vals[0])]

    print(f"\n >> Top-{topk} tokens after removing subspaces {sorted(remove_indices)} ({weight_type}):")
    for token, score in tokens_scores:
        print(f"  {token}: {score:.6f}")

    return tokens_scores


def enhanced_subspaces_proj(
    model,
    tokenizer,
    U,
    S,
    Vh,
    layer_io,
    layer_idx,
    weight_type="c_proj",
    W_E=None,
    reshape_W_E=None,
    enhance_indices=None,
    enhance_scale=2.0,
    topk=20,
):
    if enhance_indices is None:
        enhance_indices = []
    enhance_indices = set(enhance_indices)

    print(f"\nEnhancing principal components {sorted(enhance_indices)} ×{enhance_scale:.2f} ({weight_type}, layer {layer_idx})...")

    W = get_mlp_weight(model, layer_idx, weight_type)

    device = next(model.parameters()).device
    U, S, Vh = U.to(device), S.to(device), Vh.to(device)
    W = W.to(device)

    W_modified = torch.zeros_like(W)
    for i in range(S.shape[0]):
        scale = enhance_scale if i in enhance_indices else 1.0
        W_modified += scale * S[i] * torch.outer(U[:, i], Vh[i, :])

    x = layer_io["input"].to(device)
 

    # Correct projection path
    if weight_type == "c_fc":
        y_modified = x @ W_modified.T
        logits_proj = y_modified @ reshape_W_E.T
    else:
        y_modified = x @ W_modified 
        
        W_E = W_E.to(device)
        logits_proj = y_modified @ W_E.T

    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1)
    tokens_scores = [(tokenizer.decode([idx.item()]), score.item()) for idx, score in zip(top_idx[0], top_vals[0])]

    print(f"\n >> Top-{topk} tokens after enhancing subspaces {sorted(enhance_indices)} ({weight_type}):")
    for token, score in tokens_scores:
        print(f"  {token}: {score:.6f}")

    return tokens_scores
