import os
import torch
from svd_probe import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_model_analysis(model, tokenizer, text, layer_idx, device):

    # Compute SVD for the target MLP layer
    print(f"\n>> Computing SVD for Layer {layer_idx}")
    U, S, Vh = compute_svd(layer_idx, model)
    
    # Generate next token prediction
    print("\n>> Generating next token")
    next_token, next_token_id = generate_next_token(model, tokenizer, text, device)
    
    # Collect MLP layer activations
    print(f"\n>> Collecting layer-{layer_idx} input/output activations")
    layer_io, input_ids = collect_layer_input_output(model, tokenizer, layer_idx, text, device)
    
    return U, S, Vh, layer_io 


def analyze_detectors_and_effectors(
    model, 
    tokenizer, 
    U, S, Vh, 
    layer_idx, 
    layer_io,
    target_word=None, 
    topk_subspaces=10, 
    svd_ana_mode="detector_effector"
):
    """
    Unified analysis function for MLP subspaces:
      - "only_detector": analyze detector subspaces
      - "only_effector": analyze effector subspaces
      - "detector_effector": analyze full DeEf circuit
    """
    
    if svd_ana_mode == "only_detector":
        print("\n>> Detector analysis")
        top_directions = find_top_aligned_detectors(
            model, U, S, Vh, layer_idx, layer_io, topk_subspaces
        )

    elif svd_ana_mode == "only_effector":
        if target_word is None:
            raise ValueError("target_word must be provided for effector analysis.")
        print("\n>> Effector analysis")
        top_directions = find_top_aligned_effectors(
            model, tokenizer, U, S, Vh, layer_idx,
            target_word=target_word, topk_subspaces=topk_subspaces
        )

    elif svd_ana_mode == "detector_effector":
        if target_word is None:
            raise ValueError("target_word must be provided for DeEf circuit analysis.")
        print("\n>> DeEf circuit analysis")
        top_directions = find_top_DeEf_circuit(
            model, tokenizer, U, S, Vh, layer_idx,
            layer_io, target_word, topk_subspaces
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: 'only_detector', 'only_effector', or 'detector_effector'.")

    return top_directions



def subspace_interv(
    model,
    tokenizer,
    U, S, Vh,
    layer_io,
    layer_idx,
    interv_mode="enhance",
    interv_dir_indices=[6],
    interv_scale=0.8,
    return_toptoks=20
):
    """
    Intervene on MLP subspace directions to study causal effects.
        - "enhance": amplify the effect of specific subspace directions
            by scaling them with `interv_scale`.
        - "remove": suppress or eliminate certain subspace components.
    """
    # Project original output to vocab space (for comparison)
    ori_top_tokens_list = project_mlp_acv_to_vocab(
        model, tokenizer, layer_io["output"], topk=return_toptoks
    )

    # Apply subspace intervention
    if interv_mode == "remove":
        print(f"\n>> Removing subspace directions {interv_dir_indices}")
        print(" Effect: Suppresses certain semantic components in the MLP output.")
        top_tokens_list = removed_subspaces_proj(
            model, tokenizer, U, S, Vh, layer_io, layer_idx,
            remove_indices=interv_dir_indices, topk=return_toptoks
        )

    elif interv_mode == "enhance":
        print(f"\n>> Enhancing subspace directions {interv_dir_indices} (scale={interv_scale})")
        print(" Effect: Amplifies the semantic features associated with those subspaces.")
        top_tokens_list = enhanced_subspaces_proj(
            model, tokenizer, U, S, Vh, layer_io, layer_idx,
            enhance_indices=interv_dir_indices, enhance_scale=interv_scale, topk=return_toptoks
        )

    else:
        raise ValueError("interv_mode must be either 'enhance' or 'remove'.")

    return top_tokens_list


def main():
    model_name = "gpt2-medium"
    text = "The cat looks very"
    target_word = " happy"
    layer_idx = 16
    topk_subspaces = 20
    topk_tokens = 15
    output_dir = "result_probe"
    svd_ana_mode="detector_effector"  
    interv_mode = "enhance"  

    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_model(model_name, device)
    
    # MLP SVD decomposition and collection of layer activations
    U, S, Vh, layer_io = prepare_model_analysis(
        model, 
        tokenizer, 
        text, 
        layer_idx, 
        device
    )
    
    # Detector/Effector/Circuit analysis
    analyze_detectors_and_effectors(
        model, 
        tokenizer, 
        U, S, Vh, 
        layer_idx, 
        layer_io, 
        target_word, 
        topk_subspaces, 
        svd_ana_mode=svd_ana_mode
    )

    # Compute top tokens for each subspace
    layer_svd_activations, top_tokens_per_subspace = compute_subspace_top_tokens(
        model, 
        tokenizer, 
        U, S, Vh, 
        layer_io, 
        layer_idx,
        topk_subspaces=2, 
        topk_tokens=10, 
        output_dir=output_dir
    )

    # Subspace intervention
    subspace_interv(
        model, 
        tokenizer, 
        U, S, Vh, 
        layer_io, 
        layer_idx, 
        interv_mode=interv_mode, 
        interv_dir_indices=[0], 
        interv_scale=0.8, 
        return_toptoks=20
    )
    
    print("======== All Executions Completed Successfully =========")
    
if __name__ == "__main__":
    main()
