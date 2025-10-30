import argparse
import torch
import os
from circuit.circuit_interv.extract_only_circuit import extract_circuit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-2 MLP subspace intervention experiment")
    
    parser.add_argument("--mode", type=str, default="general", choices=["general", "ablation"],
                        help="Intervention mode: 'positive' for enhancing positive subspaces, 'ablation' for removing them")
    parser.add_argument("--weight_type", type=str, default="c_fc", choices=["c_fc", "c_proj"],
                        help="Type of MLP linear weight to modify")
    parser.add_argument("--top_subspaces", type=int, default=10,
                        help="Number of top subspaces to select per layer")
    parser.add_argument("--use_positive_only", action="store_true",
                        help="Only include subspaces with positive contributions")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to JSON file containing subspace results")
    parser.add_argument("--model_name", type=str, default="gpt2-medium",
                        help="HuggingFace model name (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 18, 19, 20],
                        help="List of MLP layer indices to intervene")
    parser.add_argument("--input_text", type=str, default="The cat looks very",
                        help="Input text prompt for inference")
    return parser.parse_args()


 
if __name__ == "__main__":
    args = parse_args()

    extract_circuit(
        model_name=args.model_name,
        mode=args.mode,
        weight_type=args.weight_type,
        top_subspaces=args.top_subspaces,
        use_positive_only=args.use_positive_only,
        json_file=args.json_file,
        selected_layers=args.layers,
        input_text=args.input_text,
        device=device
    )

    print("\n=== All Executions Completed Successfully ===")