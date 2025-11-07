import torch


def compute_svd(weight_type: str, c_fc=None, c_proj=None, ln_2=None):
    """
    Compute the Singular Value Decomposition (SVD) of an MLP weight matrix.
    """
    if weight_type == "c_proj":
        w = c_proj.weight

    elif weight_type == "c_fc":
        w = c_fc.weight.T.detach()  
        ln_2_weight = ln_2.weight.detach() 
        w = w * ln_2_weight 
        
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    device=w.device
    U, S, V = torch.linalg.svd(w.cpu(), full_matrices=False)
    U, S, V = U.to(device), S.to(device), V.to(device)
    
    return U, S, V
 


def reshape_emb_matrix(W_emb: torch.Tensor, c_fc, ln_2, act, use_activation: bool = False):
    """
    Project the embedding matrix through the MLP input weights (`c_fc`)
    and optionally apply the MLP activation function.
    """
    with torch.no_grad():
        # Linear transform 
        W_emb = W_emb * ln_2.weight.detach()  
        reshape_matrix = W_emb @ c_fc.weight + c_fc.bias

        # Optional nonlinearity
        if use_activation:
            reshape_matrix = act(reshape_matrix)

    reshape_matrix = reshape_matrix.to("cpu")

    return reshape_matrix

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
