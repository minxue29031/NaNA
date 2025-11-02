import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_embeddings(model_name: str, device: str | None = None):
    """
    Load a pretrained language model along with its tokenizer and input embedding matrix.
    """
 
    # Load tokenizer and model    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Extract input embedding matrix
    W_emb = model.get_input_embeddings().weight.detach().to(device)

    return model, tokenizer, W_emb


def get_mlp_matrices(model, layer_idx: int):
    """
    Extract the MLP projection matrices and activation function from a specific transformer layer.

    Returns:
        tuple:
            - c_fc (torch.nn.Linear): Feedforward input projection layer.
            - c_proj (torch.nn.Linear): Feedforward output projection layer.
            - act (callable): Activation function used in the MLP (e.g. GELU).
    """
    try:
        block = model.transformer.h[layer_idx]
    except AttributeError:
        raise AttributeError(
            "Model does not have `model.transformer.h` — "
        )

    return block.mlp.c_fc, block.mlp.c_proj, block.ln_2, block.mlp.act

def parse_topk_subspaces(topk_subspaces, total_subspaces):
    """
    Convert topk_subspaces argument to a list of indices and a label string.

    Args:
        topk_subspaces: int, list[int], or str ("all" or "topN")
        total_subspaces: int, total number of subspaces available

    Returns:
        indices: list of valid subspace indices
        label: string for file naming or display
    """
    if isinstance(topk_subspaces, str):
        if topk_subspaces.lower() == "all":
            indices = list(range(total_subspaces))
            label = "all"
        elif topk_subspaces.lower().startswith("top"):
            try:
                n = int(topk_subspaces[3:])
                indices = list(range(min(n, total_subspaces)))
                label = f"top{len(indices)}"
            except ValueError:
                raise ValueError(f"Invalid topk_subspaces format: {topk_subspaces}")
        else:
            raise ValueError(f"Invalid string for topk_subspaces: {topk_subspaces}")
    elif isinstance(topk_subspaces, int):
        indices = list(range(min(topk_subspaces, total_subspaces)))
        label = f"top{len(indices)}"
    elif isinstance(topk_subspaces, (list, tuple)):
        indices = [i for i in topk_subspaces if 0 <= i < total_subspaces]
        if not indices:
            raise ValueError("topk_subspaces list is empty or out of range.")
        label = "[" + "_".join(str(i) for i in indices) + "]"
    else:
        raise TypeError("topk_subspaces must be int, list[int], or 'all'.")

    return indices, label
