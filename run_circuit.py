import os
import torch
import argparse
import json
from block_interp.interp_mlp import load_model_and_embeddings, get_mlp_matrices
from block_interp.mlp_svd_utils import reshape_emb_matrix, compute_svd
from circuit.model_interface import generate_next_token, collect_layer_input_output
from circuit.circuit_analysis import analyze_mlp_subspaces
from circuit.collect_circuit_info import save_circuit_info
 
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 

def parse_args():
    parser = argparse.ArgumentParser(description="MLP Subspace Circuit Analysis")

    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Model name")
    parser.add_argument("--in_seq", type=str, default="The cat looks very", help="Input text")
    parser.add_argument("--target_word", type=str, default=" happy", help="Target word for analysis")
    parser.add_argument("--layers", type=str, default="16", help="MLP layer index, list or 'all'")
    parser.add_argument("--topk_subspaces", type=int, default=15, help="Top-K subspaces to analyze")
    parser.add_argument("--topk_tokens", type=int, default=20, help="Top-K tokens for effect analysis")
    parser.add_argument("--output_dir", type=str, default="result/circuit", help="Output directory")
    parser.add_argument("--circuit_mode", type=str, default="DeEf", choices=["DeEf", "De", "Ef"], help="Circuit mode")
    parser.add_argument("--interp_type", type=str, default="all", help="Interpretation type")
    parser.add_argument("--weight_type", type=str, default="c_fc", choices=["c_fc", "c_proj"],  help="Weight type for SVD")
    parser.add_argument("--return_heatmap", action="store_true", help="retrun heatmap of each directions")
                      
    parser.add_argument("--size_scale", type=float, default=200.0, help="Size scale for circuit plot")
    parser.add_argument("--color_threshold", type=float, default=2.0, help="Color threshold for circuit plot")
    parser.add_argument("--box_width", type=float, default=0.7, help="Box width for circuit plot")
                                              
    return parser.parse_args()


def prepare_model_analysis(model, tokenizer, in_seq, layer_idx, device, weight_type, c_fc, c_proj, ln_2):
    U, S, Vh = compute_svd(weight_type, c_fc, c_proj, ln_2)
    next_token, next_token_id = generate_next_token(model, tokenizer, in_seq, device)
    layer_io, input_ids = collect_layer_input_output(model, tokenizer, layer_idx, in_seq, device, weight_type)

    layer_io["input"] = layer_io["input"].to(device)
    layer_io["output"] = layer_io["output"].to(device)

    return U, S, Vh, layer_io

 


def run_mlp_circuit(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer, W_E = load_model_and_embeddings(args.model_name, device)
    
    if args.layers.lower() == "all":
        total_layers = len([m for m in model.transformer.h if hasattr(m.mlp, "c_fc")])
        layers_to_use = list(range(total_layers))
    else:
        layers_to_use = [int(x) for x in args.layers.replace(",", " ").split()]
    
    all_layers_circuits = {}
    all_cirpoints_scores = {}
    
    for layer_idx in layers_to_use:
        print(f"\n===== Processing Layer {layer_idx} =====")
        
        c_fc, c_proj, ln_2, act = get_mlp_matrices(model, layer_idx)
        reshape_W_E = reshape_emb_matrix(W_E, c_fc, ln_2, act, use_activation=False)

        # MLP SVD decomposition and collection of layer activations
        U, S, Vh, layer_io = prepare_model_analysis(
            model,
            tokenizer,
            args.in_seq,
            layer_idx,
            device,
            args.weight_type,
            c_fc,
            c_proj,
            ln_2
        )

        # Detector/Effector/Circuit analysis
        layer_circuit, circuit_point_score = analyze_mlp_subspaces(
            args.model_name,
            tokenizer,
            W_E,
            reshape_W_E,
            U,
            S,
            Vh,
            layer_idx,
            layer_io,
            args.target_word,
            args.topk_subspaces,
            args.weight_type,
            args.circuit_mode,
            args.interp_type,
            args.output_dir,
            device,
            args.topk_tokens,
            args.return_heatmap
        )
        
        all_layers_circuits[f"layer_{layer_idx}"] = layer_circuit
        all_cirpoints_scores[f"layer_{layer_idx}"] = circuit_point_score


    # Save the results including circuits and scores
    save_circuit_info(
        args.output_dir, 
        args.model_name, 
        args.weight_type, 
        args.circuit_mode, 
        all_layers_circuits, 
        all_cirpoints_scores,   
        args.size_scale, 
        args.color_threshold, 
        args.box_width
    )

    print("=== All Executions Completed Successfully ===")


if __name__ == "__main__":
    args = parse_args()
    run_mlp_circuit(args)
