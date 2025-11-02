import torch
import os


def compute_subspace_out_acv(layer_io, layer_idx, U, S, Vh, topk_subspaces):
    num_subspaces = min(topk_subspaces, S.shape[0])
    batch_size, in_dim = layer_io["input"].shape
    out_dim = U.shape[0]

    layer_svd_activations = []
    for i in range(num_subspaces):
        Wi = S[i] * torch.outer(U[:, i], Vh[i, :])  
        act_i = layer_io["input"] @ Wi  
        layer_svd_activations.append(act_i)

    return torch.stack(layer_svd_activations, dim=0)  



def subspace_acv_to_vocab(model, tokenizer, layer_svd_activations, topk_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    W_E = model.get_input_embeddings().weight.detach().cpu()
    num_subspaces, batch_size, hidden_dim = layer_svd_activations.shape
    top_tokens_per_subspace = []

    for i in range(num_subspaces):
        subspace_vec = layer_svd_activations[i]   
        logits_proj = subspace_vec @ W_E.T
        top_vals, top_idx = torch.topk(logits_proj, topk_tokens, dim=-1)

        tokens_scores = []
        for t_idx, t_score in zip(top_idx[0], top_vals[0]):
            token_str = tokenizer.decode([t_idx.item()])
            tokens_scores.append((token_str, t_score.item()))
        top_tokens_per_subspace.append(tokens_scores)

    for i, tokens_scores in enumerate(top_tokens_per_subspace):
        print(f"\nhigh-related tokens in each Subspace {i+1} for input sequence:")
        for token, score in tokens_scores:
            print(f"  {token}: {score:.6f}")


    tok_path = os.path.join(output_dir, f"top_tokens_persubspaces.tok")
    with open(tok_path, "w", encoding="utf-8") as f:
        for i, tokens_scores in enumerate(top_tokens_per_subspace):
            f.write(f">>> Subspace {i+1}:\n")
            for token, score in tokens_scores:
                f.write(f"  {token}: {score:.6f}\n")
    print(f"\n Saved top tokens per subspace to {tok_path}")

    token_to_subspace = {}
    for subspace_idx, tokens_scores in enumerate(top_tokens_per_subspace):
        for token_str, score in tokens_scores:
            if token_str not in token_to_subspace:
                token_to_subspace[token_str] = []
            token_to_subspace[token_str].append((subspace_idx, score))

    pt_path = os.path.join(output_dir, f"top_tokens_persubspaces.pt")
    torch.save(token_to_subspace, pt_path)
    print(f"Saved token-to-subspace mapping to {pt_path}")


    return top_tokens_per_subspace



def compute_subspace_top_tokens(model, tokenizer, U, S, Vh, layer_io, layer_idx, topk_subspaces=2, topk_tokens=10, output_dir="result_probe"):

    # Compute activations per subspace
    layer_svd_activations = compute_subspace_out_acv(
        layer_io, layer_idx, 
        U, S, Vh, 
        topk_subspaces
    )

    # Map activations to top tokens in the vocabulary
    top_tokens_per_subspace = subspace_acv_to_vocab(
        model, tokenizer, layer_svd_activations,
        topk_tokens, output_dir
    )

    return layer_svd_activations, top_tokens_per_subspace


