import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from block_interp.interp_mlp import load_model_and_embeddings 

# Load subspace indices from JSON
def load_subspace_indices(json_file, top_subspaces=100, use_positive_only=True):
    with open(json_file, "r") as f:
        data = json.load(f)

    layer_subspaces = {}
    for _, layer_info in data.items():
        if "subspace_results" not in layer_info:
            continue

        layer_idx = int(layer_info["layer_idx"])
        all_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"]]

        if use_positive_only:
            pos_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"] if s["contribution"] > 0]
        else:
            pos_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"] if s["contribution"]]

        selected = pos_indices[:top_subspaces]

        if selected:
            layer_subspaces[layer_idx] = selected

    print("Loaded subspace indices (preview up to 10 layers):")
    for k, v in list(layer_subspaces.items())[:10]:
        print(f"Layer {k}: {v[:10]}")
    return layer_subspaces



# Reconstruct weight from selected SVD subspaces
def reconstruct_subspace(W, indices):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_new = torch.zeros_like(W)
    for i in indices:
        if i < len(S):
            W_new += S[i] * torch.outer(U[:, i], Vh[i, :])
    return W_new


# Create a forward hook for MLP layer replacement
def make_mlp_hook(model, layer_idx, mode, weight_type, layer_subspaces, device):
    linear_module = getattr(model.transformer.h[layer_idx].mlp, weight_type)
    W_orig = linear_module.weight.data

    if layer_idx not in layer_subspaces:
        return None

    W_sub  = reconstruct_subspace(W_orig, layer_subspaces[layer_idx]).to(device)

    if mode == "general":
        W_new = W_sub
    elif mode == "ablation":
        W_new = W_orig.to(device) - W_sub
    else:
        raise ValueError("mode must be 'general' or 'ablation'")
    
    print(f"\nLayer {layer_idx}: replaced {weight_type} with {len(layer_subspaces[layer_idx])} subspaces (mode={mode})")


    def hook(module, input, output):
        x = input[0]
        if x.dim() == 3:
            x = x[:, -1, :] 
        return x @ W_new

    return hook


# Register hooks for selected layers
def register_hooks(model, mode, selected_layers, weight_type, layer_subspaces, device):
    hooks = []
    for lidx in selected_layers:
        hook_fn = make_mlp_hook(model, lidx, mode, weight_type, layer_subspaces, device)
        if hook_fn:
            target_module = getattr(model.transformer.h[lidx].mlp, weight_type)
            h = target_module.register_forward_hook(hook_fn)
            hooks.append(h)
            print(f">> Registered hook: layer {lidx} ({weight_type})")
    return hooks


# Run inference and print top-k predictions
def run_inference(model, tokenizer, text, device, topk=10):

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_last = outputs.logits[:, -1, :]
        top_vals, top_idx = torch.topk(logits_last, topk, dim=-1)

    tokens = [tokenizer.decode([i.item()]) for i in top_idx[0]]
    scores = top_vals[0].tolist()

    print(f"\nInput: {text}")
    print(f"Top-{topk} predictions:")
    for i, (tok, score) in enumerate(zip(tokens, scores), 1):
        print(f"{i:2d}. {tok!r} (logit={score:.4f})")




def extract_circuit( 
    model_name,
    mode,     
    weight_type,
    top_subspaces,
    use_positive_only,
    json_file,
    selected_layers,
    input_text,
    device):
    
    print(f"Running mode={mode}, weight_type={weight_type}, use_positive_only={use_positive_only}")

    layer_subspaces = load_subspace_indices(json_file, top_subspaces, use_positive_only)

    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)
    hooks = register_hooks(model, mode, selected_layers, weight_type, layer_subspaces, device)
    
    print("\n--- Subspace-Modified Model ---")
    run_inference(model, tokenizer, input_text, device)

    for h in hooks:
        h.remove()

    # Run baseline inference
    print("\n--- Original Model ---")
    run_inference(model, tokenizer, input_text, device)

