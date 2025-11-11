import torch
from block_interp.model_load import get_mlp_matrices
from block_interp.mlp_svd_utils import compute_svd

class MLPSubspaceManipulator:
    def __init__(self, W, layer_idx, weight_type):
        self.W = W
        self.layer_idx = layer_idx
        self.weight_type = weight_type
        
        self.c_fc, self.c_proj, self.ln_2, self.act = get_mlp_matrices(W, layer_idx)
        self.U, self.S, self.Vh = compute_svd(weight_type, c_fc=self.c_fc, c_proj=self.c_proj, ln_2=self.ln_2)
        
        if weight_type == "c_proj":
            self.W_orig = self.c_proj.weight.detach()
        elif weight_type == "c_fc":
            self.W_orig = (self.c_fc.weight.detach().T * self.ln_2.weight.detach()).T
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
            
    
    def rebuild_subspace(self, subspace_indices):
        W_reconst = torch.zeros_like(self.W_orig)
        
        for i in subspace_indices:
            if i >= len(self.S):
                continue
            if self.weight_type == "c_proj":
                W_reconst += self.S[i] * torch.outer(self.U[:, i], self.Vh[i, :])
            else:  # c_fc
                W_reconst += (self.S[i] * torch.outer(self.U[:, i], self.Vh[i, :])).T
                
        return W_reconst, self.W_orig, self.c_fc, self.c_proj, self.ln_2, self.act
        
 
    def interv_subspace(self, subspace_indices, interv_factor=1.5):
        W_interv = self.W_orig.clone()
        
        for i in subspace_indices:
            if i >= len(self.S):
                continue
            s_value = self.S[i] * (interv_factor - 1)
            if self.weight_type == "c_proj":
                W_interv += s_value * torch.outer(self.U[:, i], self.Vh[i, :])
            else:
                W_interv += (s_value * torch.outer(self.U[:, i], self.Vh[i, :])).T
        
        return W_interv, self.W_orig, self.c_fc, self.c_proj, self.ln_2, self.act
 