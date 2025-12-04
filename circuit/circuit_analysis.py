import os
import json
import torch
from block_interp.interp_mlp import MLP_DEEF_INTERP
from plot_utils.plot_heatmap import plot_subspace_heatmap


def compute_detector_alignments(x, U, S, Vh, weight_type="c_proj"):
    """Compute detector alignment scores for each subspace."""
    num_subspaces = S.shape[0]
    svd_scores = []
    
    if weight_type == "c_proj":
        for i in range(num_subspaces):
            U_i = S[i] * U[:, i]    
            svd_scores.append(x @ U_i)  
    elif weight_type == "c_fc":
        for i in range(num_subspaces):
            V_i = S[i] * Vh[i, :]    
            svd_scores.append(x @ V_i)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    return torch.stack(svd_scores, dim=0).squeeze(-1)   

def compute_effector_alignments(target_id, W_E, reshape_W_E, U, Vh, S, weight_type="c_proj"):
    """Compute effector alignment scores for each subspace."""
    num_subspaces = S.shape[0]
    
    if weight_type == "c_proj":
        # Use the original embedding matrix for projection weights
        emb_target = W_E[target_id]
        alignments = torch.tensor([Vh[i, :] @ emb_target for i in range(num_subspaces)])
    elif weight_type == "c_fc":
        # Use reshaped embedding matrix for c_fc weights
        emb_target = reshape_W_E[target_id]
        alignments = torch.tensor([U[:, i] @ emb_target for i in range(num_subspaces)])
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
        
    return alignments


def analyze_mlp_subspaces(
    model_name,
    tokenizer=None,
    W_E=None,
    reshape_W_E=None,
    U=None,
    S=None,
    Vh=None,
    layer_idx=0,
    layer_io=None,
    target_word=None,
    topk_subspaces=10,
    weight_type="c_proj",
    mode="DeEf",
    interp_type=None,
    output_dir="result",
    device=None,
    topk_tokens=20,
    return_heatmap=True, 
    do_interp=True,
    use_abs_contribute=False,
):
    """
    General function to analyze MLP subspaces.
    
    Modes:
        "De"   -> Analyze detector only
        "Ef"   -> Analyze effector only
        "DeEf" -> Analyze both detector and effector, 
                  and compute directional contribution

    Returns:
        interp_circuit: MLP_DEEF_INTERP results
        circuit_point_score: Dict with subspace contributions and detector/effector values
    """

    circuit_point_score = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "mode": mode,
        "weight_type": weight_type,
        "target_word": target_word,
        "topk_subspaces": topk_subspaces,
        "use_abs_contribute": use_abs_contribute,
        "subspace_results": []
    }

    # Detector Scores
    detector_alignments = None
    if mode in ["De", "DeEf"]:
        x = layer_io["input"]
        detector_alignments = compute_detector_alignments(
            x, U, S, Vh, weight_type
        )

    # Effector Scores
    effector_alignments = None
    if mode in ["Ef", "DeEf"]:
        target_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
        print(f"target_word '{target_word}' -> target_id: {target_id}")
        effector_alignments = compute_effector_alignments(
            target_id, W_E, reshape_W_E, U, Vh, S, weight_type
        )

    # Select Top-K Subspaces
    subspaces_for_interp = []
    
    def use_abs(x):
        return x.abs() if use_abs_contribute else x

    if mode == "DeEf":
        directional_contrib = detector_alignments.to(effector_alignments.device) * effector_alignments
        top_vals, top_idx = torch.topk(use_abs(directional_contrib), topk_subspaces)
        
        print(f"\nTop-{topk_subspaces} subspaces most aligned with '{target_word}':")
        for idx in top_idx:
            det_val = detector_alignments[idx].item()
            eff_val = effector_alignments[idx].item()
            contrib_val = directional_contrib[idx].item()
            
            print(f"  Subspace {idx+1}: Contribution {contrib_val:.6f} "
                  f"(Detector {det_val:.6f}, Effector {eff_val:.6f})")
                  
            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(contrib_val),
                "detector_value": float(det_val),
                "effector_value": float(eff_val)
            })
            
        subspaces_for_interp = top_idx.tolist()

    elif mode == "De":
        top_vals, top_idx = torch.topk(use_abs(detector_alignments), topk_subspaces)
        
        print(f"\nTop-{topk_subspaces} detector subspaces:")
        for idx in top_idx:
            val = detector_alignments[idx].item()
            print(f"  Subspace {idx+1}: Contribution {val:.6f}")
            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(val)
            })
            
        subspaces_for_interp = top_idx.tolist()

    elif mode == "Ef":
        top_vals, top_idx = torch.topk(use_abs(effector_alignments), topk_subspaces)
        
        print(f"\nTop-{topk_subspaces} effector subspaces:")
        for idx in top_idx:
            val = effector_alignments[idx].item()
            print(f"  Subspace {idx+1}: Contribution {val:.6f}")
            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(val)
            })
            
        subspaces_for_interp = top_idx.tolist()

    else:
        raise ValueError(f"Unknown mode: {mode}")


    # Call MLP_DEEF_INTERP
    interp_circuit = None
    if do_interp:
        interp_io = MLP_DEEF_INTERP(model_name, output_dir, device)
        with torch.no_grad():
            interp_circuit = interp_io.mlp_subspace_interp(
                layer_idx=layer_idx,
                out_dir=output_dir,
                topk_tokens=topk_tokens,
                topk_subspaces=subspaces_for_interp,
                weight_type=weight_type,
                interp_type=interp_type,
                with_negative=True,
                save_file=False,
                return_heatmap=return_heatmap 
            )
        
    
    return interp_circuit, circuit_point_score
