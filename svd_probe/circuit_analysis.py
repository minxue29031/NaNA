import torch

  
def find_top_aligned_detectors(model, U, S, Vh, layer_idx, layer_io, topk_subspaces=10):
 
    svd_scores = []
    max_k = min(topk_subspaces, S.shape[0])

    for i in range(max_k):
        U_i = S[i] * U[:, i]       
        score_i = layer_io["input"] @ U_i   
        svd_scores.append(score_i)

    svd_scores_tensor = torch.stack(svd_scores, dim=0).squeeze(-1)  
    top_vals, top_idx = torch.topk(svd_scores_tensor.abs(), topk_subspaces)

    top_aligned_subspaces = [(idx.item(), svd_scores_tensor[idx].item()) for idx in top_idx]
 
    print(f"\n Top-{topk_subspaces} active detectors with original activation values:")
    for idx, val in top_aligned_subspaces:
        print(f"  Subspace_detector {idx+1}: activation {val:.6f}")


    return top_aligned_subspaces

def find_top_aligned_effectors(model, tokenizer, U, S, Vh, layer_idx, target_word, topk_subspaces=10):

    # target token embedding
    target_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    
    print(f"target_word {target_word} -> target_id: {target_id}")
    W_E = model.get_input_embeddings().weight.detach().cpu()
    emb_target = W_E[target_id]  
 
    alignments = Vh @ emb_target  
    top_vals, top_idx = torch.topk(alignments.abs(), topk_subspaces)

    top_aligned_subspaces = [(idx.item(), alignments[idx].item()) for idx in top_idx]

    print(f"\nTop-{topk_subspaces} subspaces most aligned with '{target_word}' (original values):")
    for idx, score in top_aligned_subspaces:
        print(f"  Subspace_effector {idx+1}: alignment {score:.6f}")

    return top_aligned_subspaces

def find_top_DeEf_circuit(model, tokenizer, U, S, Vh, layer_idx, layer_io, target_word, topk_subspaces=10):
 
    x = layer_io["input"]  
    
    # Detector
    svd_scores = []
    num_subspaces = min(topk_subspaces, S.shape[0])
    for i in range(num_subspaces):
        U_i = S[i] * U[:, i]
        score_i = x @ U_i
        svd_scores.append(score_i)

    svd_scores_tensor = torch.stack(svd_scores, dim=0).squeeze(-1) 

    # Effector
    target_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    print(f"target_word {target_word} -> target_id: {target_id}")
    W_E = model.get_input_embeddings().weight.detach().cpu()
    e_target = W_E[target_id]

    effector_alignments = torch.zeros(num_subspaces)
    for i in range(num_subspaces):
        effector_alignments[i] = (Vh[i] @ e_target) 

    directional_contrib = svd_scores_tensor * effector_alignments  
    
    print(f"\nTop-{topk_subspaces} subspaces for input '{target_word}':")
    for i in range(num_subspaces):
        print(f"  Subspace {i+1} Contribution {directional_contrib[i]:.6f} --> " 
              f"Detector ({svd_scores_tensor[i]:.6f}) + Effector ({effector_alignments[i]:.6f})")

    # top-aligned subspaces
    top_vals, top_idx = torch.topk(directional_contrib.abs(), topk_subspaces)
    top_aligned_subspaces = []
    print(f"\nTop-{topk_subspaces} subspaces most aligned with '{target_word}':")
    for idx in top_idx:
        det_val = svd_scores_tensor[idx].item()
        eff_val = effector_alignments[idx].item()
        contrib_val = directional_contrib[idx].item()
        print(f"  Subspace {idx+1} Contribution {contrib_val:.6f} --> "
              f"Detector ({det_val:.6f}) + Effector ({eff_val:.6f})")
        top_aligned_subspaces.append({
            "subspace": idx.item()+1,
            "alignment": contrib_val,
            "detector": det_val,
            "effector": eff_val
        })

    return top_aligned_subspaces
