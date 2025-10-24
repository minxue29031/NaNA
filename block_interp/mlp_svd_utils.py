import torch


def compute_svd(weight_type: str, c_fc=None, c_proj=None, ln_2=None):
    """
    Compute the Singular Value Decomposition (SVD) of an MLP weight matrix.
    """
    
    if weight_type == "c_proj":
        if c_proj is None:
            raise ValueError("`c_proj` must be provided when weight_type='c_proj'")
        w = c_proj.weight

    elif weight_type == "c_fc":
        if c_fc is None:
            raise ValueError("`c_fc` must be provided when weight_type='c_fc'")
 
        w = c_fc.weight.T.detach()  
        ln_2_weight = ln_2.weight.detach() 
        w = w * ln_2_weight 

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
 
    U, S, V = torch.linalg.svd(w, full_matrices=False)
    return U, S, V
 


def reshape_emb_matrix(W_emb: torch.Tensor, c_fc, act, use_activation: bool = False):
    """
    Project the embedding matrix through the MLP input weights (`c_fc`)
    and optionally apply the MLP activation function.
    """
    
    # Linear transform 
    reshape_matrix = W_emb @ c_fc.weight + c_fc.bias

    # Optional nonlinearity
    if use_activation:
        reshape_matrix = act(reshape_matrix)

    return reshape_matrix
