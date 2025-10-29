import torch
 

def removed_subspaces_proj(model, tokenizer, U, S, Vh, layer_io, layer_idx, remove_indices=None, topk=20):
    if remove_indices is None:
        remove_indices = []

    remove_indices = set(remove_indices)
    print(f"\nRemoving principal components {sorted(remove_indices)} from MLP projection (layer {layer_idx})...")

    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu()  
 

    W_reduced = torch.zeros_like(weight)
    for i in range(S.shape[0]):
        if i in remove_indices:
            print(">> remove_indices", i)
            continue
        W_reduced += S[i] * torch.outer(U[:, i], Vh[i, :])

    x = layer_io["input"]  
    y_reduced = x @ W_reduced 

    W_E = model.get_input_embeddings().weight.detach().cpu() 
    logits_proj = y_reduced @ W_E.T  

    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1)
    tokens_scores = []
    for idx, score in zip(top_idx[0], top_vals[0]):
        token_str = tokenizer.decode([idx.item()])
        tokens_scores.append((token_str, score.item()))

    print(f"\n >> Top-{topk} tokens after removing SVD-based subspaces {sorted(remove_indices)}:")
    for token, score in tokens_scores:
        print(f"  {token}: {score:.6f}")

    return tokens_scores


def enhanced_subspaces_proj(model, tokenizer, U, S, Vh, layer_io, layer_idx, enhance_indices=None, enhance_scale=2.0, topk=20):
    if enhance_indices is None:
        enhance_indices = []

    enhance_indices = set(enhance_indices)
    print(f"\nEnhancing principal components {sorted(enhance_indices)} by ×{enhance_scale:.2f} (layer {layer_idx})...")

    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu() 

    # Reconstruct the enhanced matrix
    W_modified = torch.zeros_like(weight)
    for i in range(S.shape[0]):
        scale = enhance_scale if i in enhance_indices else 1.0
        W_modified += scale * S[i] * torch.outer(U[:, i], Vh[i, :])

    # Get layer input and calculate projection output
    x = layer_io["input"]  
    y_modified = x @ W_modified   
    W_E = model.get_input_embeddings().weight.detach().cpu() 
    
    row = W_E[1]  
    norm = row.norm().item()
    dim = row.shape
    
    logits_proj = y_modified @ W_E.T  

    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1)
    tokens_scores = []
    for idx, score in zip(top_idx[0], top_vals[0]):
        token_str = tokenizer.decode([idx.item()])
        tokens_scores.append((token_str, score.item()))

    print(f"\n >> Top-{topk} tokens after enhancing SVD-based subspaces {sorted(enhance_indices)}:")
    for token, score in tokens_scores:
        print(f"  {token}: {score:.6f}")

    return tokens_scores  