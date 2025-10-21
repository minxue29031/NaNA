import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def collect_layer_input_output(model, tokenizer, layer_idx, text):
    layer_inputs, layer_outputs = [], []

    def hook_fn(module, input, output):
        layer_inputs.append(input[0][:, -1, :].detach().cpu())
        layer_outputs.append((output[:, -1, :].detach().cpu()))

    block = model.transformer.h[layer_idx]
    hook = block.mlp.c_proj.register_forward_hook(hook_fn)

    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    hook.remove()

    layer_input = torch.cat(layer_inputs, dim=0)
    layer_output = torch.cat(layer_outputs, dim=0)

    print(f"Layer {layer_idx} input shape: {layer_input.shape}")
    print(f"Layer {layer_idx} output shape: {layer_output.shape}")

    return {"input": layer_input, "output": layer_output}, inputs["input_ids"]

 

def compute_svd_activations_any(layer_io, layer_idx, model, topk_subspaces):
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu()  
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
 
    num_subspaces = min(topk_subspaces, S.shape[0])
    batch_size, in_dim = layer_io["input"].shape
    out_dim = U.shape[0]

    layer_svd_activations = []
    for i in range(num_subspaces):
        Wi = S[i] * torch.outer(U[:, i], Vh[i, :])  
        act_i = layer_io["input"] @ Wi  
        layer_svd_activations.append(act_i)

    return torch.stack(layer_svd_activations, dim=0)  

 

def svd_activations_to_vocab(model, tokenizer, layer_svd_activations, topk_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    W_E = model.get_input_embeddings().weight.detach().cpu()
    W_E = W_E / W_E.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12)
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
        print(f"\nSubspace {i}:")
        for token, score in tokens_scores:
            print(f"  {token}: {score:.6f}")


    tok_path = os.path.join(output_dir, f"top_tokens_persubspaces.tok")
    with open(tok_path, "w", encoding="utf-8") as f:
        for i, tokens_scores in enumerate(top_tokens_per_subspace):
            f.write(f"----------Subspace {i}----------:\n")
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
    
    
 
def compute_svd_subspace_scores(model, layer_idx, layer_io, k=1024, topk=50):
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu()   
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    svd_scores = []
    avg_abs_scores = []

    max_k = min(k, S.shape[0])

    for i in range(max_k):
        U_i = S[i] * U[:, i]  
        score_i = layer_io["input"] @ U_i   
        svd_scores.append(score_i)
        avg_abs_scores.append(score_i.abs().item())

    svd_scores_tensor = torch.cat(svd_scores, dim=0)  
    avg_abs_scores_tensor = torch.tensor(avg_abs_scores)  
    top_vals, top_idx = torch.topk(avg_abs_scores_tensor, topk)
    
 
    print(f"\nTop-{topk} active subspaces and their original scores:")
    for idx in top_idx[:30]: 
        original_score = svd_scores_tensor[idx].item()
        print(f"  Subspace {idx.item()}: {original_score:.6f}")

    return svd_scores_tensor, U, S, Vh, top_idx, top_vals


def generate_next_token(model, tokenizer, text, max_new_tokens=1):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits   
        last_token_logits = logits[:, -1, :]   
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1)
        next_token = tokenizer.decode(next_token_id)
        
    print(f"\n >>>>>>> Next token: {next_token} (ID: {next_token_id.item()})")
    
    return next_token, next_token_id.item()


def project_layer_output_to_vocab(model, tokenizer, layer_output, topk=20):
    W_E = model.get_input_embeddings().weight.detach().cpu() 
    logits_proj = layer_output @ W_E.T  
    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1) 

    top_tokens_list = []
    for batch_idx in range(layer_output.shape[0]):
        tokens = [tokenizer.decode([idx.item()]) for idx in top_idx[batch_idx]]
        scores = top_vals[batch_idx].tolist()
        top_tokens_list.append(list(zip(tokens, scores)))

    print(f"\n >> Ori top-{topk_tokens} tokens by projecting MLP output:")
    for token, score in top_tokens_list[0]:
        print(f"  {token}: {score:.6f}")

    return top_tokens_list

def project_without_components(model, tokenizer, layer_io, layer_idx, remove_indices=None, topk=20):
    if remove_indices is None:
        remove_indices = []

    remove_indices = set(remove_indices)
    print(f"\nRemoving principal components {sorted(remove_indices)} from MLP projection (layer {layer_idx})...")

    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu()  
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

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


def project_with_enhanced_components(model, tokenizer, layer_io, layer_idx, enhance_indices=None, enhance_scale=2.0, topk=20):
    if enhance_indices is None:
        enhance_indices = []

    enhance_indices = set(enhance_indices)
    print(f"\nEnhancing principal components {sorted(enhance_indices)} by ×{enhance_scale:.2f} (layer {layer_idx})...")

    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    weight = c_proj.weight.detach().cpu() 
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

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


def compute_subspace_top_tokens(model, tokenizer, layer_io, layer_idx, topk_subspaces=2, topk_tokens=10, output_dir="result_probe"):

    # Compute activations per subspace
    layer_svd_activations = compute_svd_activations_any(
        layer_io, layer_idx, model, topk_subspaces
    )

    # Map activations to top tokens in the vocabulary
    top_tokens_per_subspace = svd_activations_to_vocab(
        model, tokenizer, layer_svd_activations,
        topk_tokens=topk_tokens,
        output_dir=output_dir
    )

    return layer_svd_activations, top_tokens_per_subspace

 

if __name__ == "__main__":
    model_name = "gpt2-medium"
    text = "The cat looks very"  
    layer_idx = 16
    topk_subspaces = 2
    topk_tokens =2
    output_dir = "result_probe"
    mode = "enhance"  # Options: "remove" or "enhance"
    
    # Load Model and collect info 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    next_token, next_token_id = generate_next_token(model, tokenizer, text)  
    layer_io, input_ids = collect_layer_input_output(model, tokenizer, layer_idx, text) 

    # Compute SVD subspace scores and get top subspaces
    svd_scores, U, S, Vh, top_idx, top_vals = compute_svd_subspace_scores(
        model, layer_idx, layer_io, k=topk_subspaces, topk=topk_tokens
    )
    
    # Compute activations for each subspace and get top tokens per subspace 
    layer_svd_activations, top_tokens_per_subspace = compute_subspace_top_tokens(
        model, tokenizer, layer_io, layer_idx=16,
        topk_subspaces=2, topk_tokens=10, output_dir="result_probe"
    )
 
    # # Project original layer output to vocabulary & Modify subspace behavior (remove or enhance)
    ori_top_tokens_list = project_layer_output_to_vocab(model, tokenizer, layer_io["output"], topk=topk_tokens)
    
    if mode == "remove":
        removed_subspaces = project_without_components(
            model, tokenizer, layer_io, layer_idx,
            remove_indices=[2, 14, 60, 11], topk=20
        )
        result = removed_subspaces

    else:
        enhanced_subspaces = project_with_enhanced_components(
            model, tokenizer, layer_io, layer_idx,
            enhance_indices=[0], enhance_scale=0.8, topk=20
        )
        result = enhanced_subspaces
