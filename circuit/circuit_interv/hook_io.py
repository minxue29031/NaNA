import os
import json
import torch
from tqdm import tqdm
from circuit.circuit_interv.reconst_mlp import MLPSubspaceManipulator
from circuit.circuit_interv.show_map import print_top_tokens
from block_interp.mlp_svd_utils import compute_svd, reshape_emb_matrix


 
def create_last_token_hook(buffer, layer_idx):
    """
    Save the input of the last token for a given layer into a buffer.
    """
    def hook(module, input, output):
        buffer[layer_idx] = input[0][:, -1:, :].detach().clone()
    return hook


 
def show_intervention(
        tokenizer, 
        x_ori,
        x_last, 
        layer_idx, 
        W_orig, 
        W_new, 
        weight_type, 
        W_E=None, 
        reshape_W_E=None, 
        bias=None,
        input_text=None,   
        save_path=None,
        token_num=10        
    ):
    """
    Compute linear transformation before and after intervention,
    print top tokens, and always save them with input text.
    """
    y_before = torch.nn.functional.linear(x_ori, W_orig.T, bias)
    y_after = torch.nn.functional.linear(x_last, W_new.T, bias)

    print(f"\n>>> {weight_type} BEFORE intervention for layer {layer_idx}:")
    top_before = print_top_tokens(tokenizer, y_before, weight_type, W_E=W_E, reshape_W_E=reshape_W_E, topk=token_num)
    
    print(f"\n>>> {weight_type} AFTER intervention for layer {layer_idx}:")
    top_after = print_top_tokens(tokenizer, y_after, weight_type, W_E=W_E, reshape_W_E=reshape_W_E, topk=token_num)

    before_list = [{"token": t.strip(), "score": float(s)} for t, s in top_before]
    after_list  = [{"token": t.strip(), "score": float(s)} for t, s in top_after]

    json_data = {
        "input_text": input_text,
        "original": before_list,
        "modified": after_list
    }

    if save_path is not None:
        all_layers_dir = os.path.join(save_path, "all_layers")
        os.makedirs(all_layers_dir, exist_ok=True)
        file_path = os.path.join(all_layers_dir, f"layer{layer_idx}_{weight_type}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    return y_before, y_after


def apply_intervention(output, y_after, use_full_residual):
    """
    Replace the last token output with intervened output
    if use_full_residual is True.
    """
    if use_full_residual:
        out = output.clone()
        out[:, -1:, :] = y_after
        return out
    else:
        return y_after



def get_modified_weights(
        manipulator, 
        subspace_indices, 
        modify_type, 
        interv_factor, 
        gene_or_abla
    ):
    """
    Return modified and original weights along with related modules
    based on modification type.
    """
    if modify_type == "rebuild":
        W_reconst, W_orig, c_fc, c_proj, ln_2, act = manipulator.rebuild_subspace(subspace_indices)
        W_new = W_reconst if gene_or_abla == "general" else (W_orig - W_reconst)
    elif modify_type in ["manual_interv", "auto_interv"]:
        W_reconst, W_orig, c_fc, c_proj, ln_2, act = manipulator.interv_subspace(subspace_indices, interv_factor)
        W_new = W_reconst
    else:
        raise ValueError(f"Unsupported modify_type: {modify_type}")
    return W_new, W_orig, c_fc, c_proj, ln_2, act




def create_subspace_hook(
        model, 
        layer_idx, 
        weight_type, 
        gene_or_abla, 
        layer_subspaces, 
        use_bias,
        ln2_inputs_last_token, 
        mlp_activations_last_token,
        ln2_inputs_clean,          
        mlp_activations_clean,  
        modify_type,
        interv_factor,
        use_full_residual,
        tokenizer,
        W_E,
        save_dir,
        input_text,
        token_num
    ):
    
    
    if layer_idx not in layer_subspaces:
        return None

    subspace_indices = layer_subspaces[layer_idx]
    manipulator = MLPSubspaceManipulator(model, layer_idx, weight_type)
    W_new, W_orig, c_fc, c_proj, ln_2, act = get_modified_weights(
        manipulator, 
        subspace_indices, 
        modify_type, 
        interv_factor, 
        gene_or_abla
    )

    # reshape embedding matrix
    reshape_W_E = reshape_emb_matrix(W_E, c_fc, ln_2, act, use_activation=True)

    # c_fc hook
    if weight_type == "c_fc":
        ln2_weight = ln_2.weight.data
        ln2_bias = ln_2.bias.data
        c_fc_weight = c_fc.weight.data
        c_fc_bias = c_fc.bias.data

        fused_bias = c_fc_bias + (ln2_bias @ c_fc_weight) if use_bias else None

        def hook(module, input, output):
            x_last = ln2_inputs_last_token[layer_idx]
            x_ori = ln2_inputs_clean[layer_idx]
            
            if x_last is None:
                return output
            _, y_after = show_intervention(
                tokenizer, 
                x_ori,
                x_last, 
                layer_idx, 
                W_orig, 
                W_new, 
                "c_fc", 
                W_E, 
                reshape_W_E, 
                bias=fused_bias,
                input_text=input_text,
                save_path=save_dir,
                token_num=token_num
            )
            return apply_intervention(output, y_after, use_full_residual)

    # c_proj hook
    elif weight_type == "c_proj":
        c_proj = getattr(model.transformer.h[layer_idx].mlp, "c_proj")
        c_proj_bias = c_proj.bias.data if use_bias else None

        def hook(module, input, output):
            x_last = mlp_activations_last_token[layer_idx]
            x_ori = mlp_activations_clean[layer_idx]
            
            if x_last is None:
                return output
            _, y_after = show_intervention(
                tokenizer, 
                x_ori,
                x_last, 
                layer_idx, 
                W_orig, 
                W_new, 
                "c_proj", 
                W_E, 
                reshape_W_E, 
                bias=c_proj_bias,
                input_text=input_text,
                save_path=save_dir,
                token_num=token_num
            )
            return apply_intervention(output, y_after, use_full_residual)

    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    return hook




def register_hooks(
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
        save_dir,
        input_text,
        token_num
    ):
    """
    Register forward hooks for last token capture and subspace interventions.
    """
    
    handles = []
    num_layers = len(model.transformer.h)
    
    ln2_inputs_clean, mlp_activations_clean = capture_clean_inputs(model, tokenizer, input_text, device )
    
    # Buffers to store original last token inputs for each layer
    ln2_inputs_last_token = [None] * num_layers
    mlp_activations_last_token = [None] * num_layers

    # Register hooks to capture original last token inputs
    for i, block in enumerate(model.transformer.h):
        block.ln_2.register_forward_hook(
            create_last_token_hook(ln2_inputs_last_token, i)
        )
        block.mlp.c_proj.register_forward_hook(
            create_last_token_hook(mlp_activations_last_token, i)
        )
    
    # Register subspace intervention hooks
    print(f"Registering hooks for {len(selected_layers)} layers...")
    for layer_idx in tqdm(selected_layers, desc="Layer Hooks"):
        hook_fn = create_subspace_hook(
            model,
            layer_idx,
            weight_type,
            gene_or_abla,
            layer_subspaces,
            use_bias,
            ln2_inputs_last_token,
            mlp_activations_last_token,
            ln2_inputs_clean,          
            mlp_activations_clean,  
            modify_type,
            interv_factor,
            use_full_residual,
            tokenizer,
            W_E,
            save_dir,
            input_text,
            token_num
        )
        if hook_fn is not None:
            target_module = getattr(model.transformer.h[layer_idx].mlp, weight_type)
            handle = target_module.register_forward_hook(hook_fn)
            handles.append(handle)

    return handles
    


def capture_clean_inputs(model, tokenizer, inputs, device, attention_mask=None):
    """
    Run a forward pass through the model to capture clean (unintervened) inputs 
    for each layer's ln_2 and mlp.c_proj last token.
    """
    
    encoded = tokenizer(inputs, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    num_layers = len(model.transformer.h)
    ln2_inputs_clean = [None] * num_layers
    mlp_activations_clean = [None] * num_layers

    temp_handles = []
    for i, block in enumerate(model.transformer.h):
        temp_handles.append(
            block.ln_2.register_forward_hook(create_last_token_hook(ln2_inputs_clean, i))
        )
        temp_handles.append(
            block.mlp.c_proj.register_forward_hook(create_last_token_hook(mlp_activations_clean, i))
        )

    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    for h in temp_handles:
        h.remove()

    return ln2_inputs_clean, mlp_activations_clean