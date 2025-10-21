import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_svd_input_similarity(model, tokenizer, layer_idx, W_emb, topk=20, k=1024, with_negative=False, use_activation=False):

    block = model.transformer.h[layer_idx]
    c_proj = block.mlp.c_proj
    c_fc = block.mlp.c_fc
    act = block.mlp.act

    # Compute the scores matrix: embedding -> MLP c_fc (optionally with activation)
    scores_matrix = W_emb @ c_fc.weight.to(device) + c_fc.bias.to(device)
    if use_activation:
        scores_matrix = act(scores_matrix)

    # Perform SVD on the c_proj weight
    U, S, Vh = torch.linalg.svd(c_proj.weight.detach().cpu(), full_matrices=False)
    print("------------U[:, 1].shape------", U[:, 1].shape)

    all_results = []
    all_pos_scores = []
    all_neg_scores = []

    for i in tqdm(range(min(k, S.shape[0])), desc=f"Layer {layer_idx} SVD similarity"):
        # Compute the direction vector
        u_i = S[i] * U[:, i]
        scores = scores_matrix @ u_i.to(device)

        # Positive tokens 
        pos_mask = scores > 0
        if pos_mask.any():
            masked_indices = torch.arange(scores.shape[0], device=device)[pos_mask]
            pos_vals, topk_idx_in_mask = torch.topk(scores[pos_mask], min(topk, pos_mask.sum()))
            pos_idx = masked_indices[topk_idx_in_mask]
            pos_tokens = [tokenizer.decode([int(idx)]) for idx in pos_idx]
        else:
            pos_tokens = ["=== Insufficient positive tokens ==="]
            pos_vals = torch.tensor([float("nan")], device=device)

        all_pos_scores.append(pos_vals.cpu())

        # Negative tokens 
        neg_mask = scores < 0
        if neg_mask.any():
            masked_indices = torch.arange(scores.shape[0], device=device)[neg_mask]
            neg_vals, topk_idx_in_mask = torch.topk(-scores[neg_mask], min(topk, neg_mask.sum()))
            neg_vals = -neg_vals
            neg_idx = masked_indices[topk_idx_in_mask]
            neg_tokens = [tokenizer.decode([int(idx)]) for idx in neg_idx]
        else:
            neg_tokens = ["=== Insufficient negative tokens ==="]
            neg_vals = torch.tensor([float("nan")], device=device)

        all_neg_scores.append(neg_vals.cpu())

        all_results.append({
            "pos": list(zip(pos_tokens, pos_vals.cpu().tolist())),
            "neg": list(zip(neg_tokens, neg_vals.cpu().tolist()))
        })

    return all_results, all_pos_scores, all_neg_scores


def save_results(output_dir, layer_idx, svd_results, topk, k, with_negative=False, print_scores=True):

    os.makedirs(output_dir, exist_ok=True)
    tok_path = os.path.join(output_dir, f"MLPout_layer{layer_idx}_top{k}subspaces_top{topk}tokens_detector.txt")

    with open(tok_path, "a", encoding="utf-8") as f:
        for i, result in enumerate(svd_results):
            # Positive 
            if print_scores:
                pos_tokens = [f"{t}({s:.6f})" for t, s in result["pos"]]
            else:
                pos_tokens = [t for t, s in result["pos"]]
            pos_line = ", ".join(pos_tokens)
            f.write(f"Direction {i+1} POS:\n{pos_line}\n\n")
            print(f"Direction {i+1} POS:")
            print(pos_line, "\n")

            # Negative 
            if with_negative:
                if print_scores:
                    neg_tokens = [f"{t}({s:.6f})" for t, s in result["neg"]]
                else:
                    neg_tokens = [t for t, s in result["neg"]]
                neg_line = ", ".join(neg_tokens)
                f.write(f"Direction {i+1} NEG:\n{neg_line}\n\n")
                print(f"Direction {i+1} NEG:")
                print(neg_line, "\n")

    print(f">> Saved: {tok_path}")


if __name__ == "__main__":

    model_name = "gpt2-medium"
    layers_to_use = [16]  
    topk_subspaces = 100
    topk_tokens = 50
    output_dir = "result"
    calc_negative = False
    print_scores = False
    use_activation4emb = True

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    W_emb = model.get_input_embeddings().weight.detach().to(device)
    print(f"Embedding shape: {W_emb.shape}")

    # Process each layer
    for layer_idx in layers_to_use:
        print(f"\n=== Processing Layer {layer_idx} ===")
        svd_results, pos_scores, neg_scores = compute_svd_input_similarity(
            model, tokenizer, layer_idx, W_emb, topk=topk_tokens, k=topk_subspaces,
            with_negative=calc_negative, use_activation=use_activation4emb
        )
        save_results(
            output_dir, layer_idx, svd_results, topk=topk_tokens, k=topk_subspaces, 
            with_negative=calc_negative, print_scores=print_scores
        )
