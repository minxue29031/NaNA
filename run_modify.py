import argparse
import torch
import os
from circuit.circuit_interv.mod_circuit import reubuld_interv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-2 MLP subspace intervention experiment")
    
    parser.add_argument("--gene_or_abla", type=str, default="general", choices=["general", "ablation"], help="Intervention gene_or_abla : 'general' for reconstruct mlp using top subspaces, 'ablation' for removing them")
    parser.add_argument("--weight_type", type=str, default="c_fc", choices=["c_fc", "c_proj"],
                        help="Type of MLP linear weight to modify")
    parser.add_argument("--top_subspaces", type=int, default=10,
                        help="Number of top subspaces to select per layer")
    parser.add_argument("--use_positive_only", action="store_true",
                        help="Only include subspaces with positive contributions")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to JSON file containing subspace results")
    parser.add_argument("--model_name", type=str, default="gpt2-medium",
                        help="HuggingFace gene_or_abla l name (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 18, 19, 20],
                        help="List of MLP layer indices to intervene")
    parser.add_argument("--use_bias", action="store_true", 
                        help="Modified MLP using bias")
    parser.add_argument("--input_text", type=str, default="The cat looks very",
                        help="Input text prompt for inference")
    parser.add_argument("--output_dir", type=str, default="result/interven_result", help="Output directory")
    
    parser.add_argument("--modify_type", type=str, default="rebuild", choices=["rebuild", "interv"],
                        help="Type of modification to apply: 'rebuild' or 'interv'")
    parser.add_argument("--interv_factor", type=float, default=0.1,
                        help="Scaling factor for intervention")
    parser.add_argument("--use_full_residual", action="store_true",
                        help="Whether to use full residual during modification")
    parser.add_argument("--token_num", type=int, default=20,
                        help="Number of top tokens to display during inference")
    
    return parser.parse_args()


 
if __name__ == "__main__":
    args = parse_args()

    with torch.no_grad():
        reubuld_interv(
            model_name=args.model_name,
            gene_or_abla =args.gene_or_abla,
            weight_type=args.weight_type,
            top_subspaces=args.top_subspaces,
            use_positive_only=args.use_positive_only,
            json_file=args.json_file,
            selected_layers=args.layers,
            input_text=args.input_text,
            use_bias=args.use_bias,
            device=device,
            modify_type=args.modify_type,
            interv_factor=args.interv_factor,
            use_full_residual=args.use_full_residual,
            token_num=args.token_num,
            output_dir=args.output_dir
        )

    print("\n=== All Executions Completed Successfully ===")
