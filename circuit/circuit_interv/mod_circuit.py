import json
import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from block_interp.model_load import get_mlp_matrices
from circuit.circuit_interv.hook_io import register_hooks
from circuit.circuit_interv.show_map import show_infer
from block_interp.model_load import load_model_and_embeddings

# Load subspace indices from JSON
def load_subspace_indices(json_file, top_subspaces=100, use_positive_only=True, use_random_index=False):
    random.seed(42)
    with open(json_file, "r") as f:
        data = json.load(f)

    layer_subspaces = {}
    for _, layer_info in data.items():
        if "subspace_results" not in layer_info:
            continue

        layer_idx = int(layer_info["layer_idx"])
        all_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"]]
        
        # Select only positive-contributing subspaces if requested
        if use_positive_only:
            pos_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"] if s["contribution"] > 0]
            print(f"[Layer {layer_idx}] Positive subspaces: {len(pos_indices)}")

        else:
            pos_indices = [s["subspace_index"] - 1 for s in layer_info["subspace_results"] if s["contribution"]]

        if use_random_index:
            selected = random.sample(pos_indices, k=min(top_subspaces, len(pos_indices)))
        else:
            selected = pos_indices[:top_subspaces]

        layer_subspaces[layer_idx] = selected

    print("Loaded subspace indices (preview up to 10 layers):")
    for k, v in list(layer_subspaces.items()):
        print(f"Layer {k}: {v}")
        
    return layer_subspaces


   

def reubuld_interv( 
    model_name,
    gene_or_abla,     
    weight_type,
    top_subspaces,
    use_positive_only,
    manual_subspace_file,
    auto_subspace_file,
    selected_layers,
    input_text, 
    use_bias,
    device,
    modify_type ="rebuild",  # options: "rebuild", "auto_interv", "manual_interv"
    interv_factor=0.1,
    use_full_residual=True,
    token_num=20,
    output_dir=None,
    use_random_index=False):
    
    print(f"Running modify type: {modify_type}, mode={gene_or_abla}, weight_type={weight_type}, use_positive_only={use_positive_only}")
 
    if modify_type == "manual_interv":
        print("Manual intervention mode: using user-provided JSON for subspace indices.")
        with open(manual_subspace_file, "r") as f:
            raw_data = json.load(f)
        layer_subspaces = {int(k): [idx - 1 for idx in v] for k, v in raw_data.items()}
    else:
        layer_subspaces = load_subspace_indices(auto_subspace_file, top_subspaces, use_positive_only, use_random_index)
    
    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)
    
    hooks = register_hooks(
        model, 
        tokenizer, 
        gene_or_abla, 
        selected_layers, 
        weight_type, 
        layer_subspaces, 
        use_bias, 
        modify_type, 
        interv_factor, 
        use_full_residual, 
        device, 
        W_E,
        save_dir=output_dir,
        input_text=input_text,
        token_num=token_num
    )
     

    final_prediction = show_infer(
        model, 
        tokenizer, 
        model_name,
        gene_or_abla,
        top_subspaces,        
        input_text, 
        hooks, 
        device, 
        topk=token_num, 
        save_dir=output_dir
    )

 
