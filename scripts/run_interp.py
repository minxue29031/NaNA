import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import argparse
from typing import List
from block_interp.interp_mlp import MLP_DEEF_INTERP
from block_interp.model_load import parse_layers_arg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

def parse_args():
    parser = argparse.ArgumentParser(description="Professional MLP Subspace Interpretation Tool")
    
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Hugging Face model name (e.g., 'gpt2-medium')")
    parser.add_argument("--layers", nargs='+', default=["all"], help="Specify layers (e.g. --layers 4 5 6) or 'all'")
    parser.add_argument("--out_dir", type=str, default="result", help="Directory to save results")
    parser.add_argument("--topk_tokens", type=int, default=10, help="Number of top tokens per direction")
    parser.add_argument("--topk_subspaces", type=int, default=50, help="Number of top singular directions to analyze")
    parser.add_argument("--weight_type", type=str, choices=["c_proj", "c_fc", "ov"], default="c_proj", help="Weight matrix type")
    parser.add_argument("--interp_type", type=str, choices=["detector", "effector", "all"], default="detector", help="Interpretation type")
    parser.add_argument("--with_negative", action="store_true", help="Save negative directions as well")
    parser.add_argument("--save_file", action="store_true", help="Save the computed subspace results to a file.")
    parser.add_argument("--return_heatmap", action="store_true", help="Return the subspace heatmap.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cuda/cpu); auto-detect if None")
    
    return parser.parse_args()



def run_mlp_analysis(
    model_name: str,
    layers: List[int],
    out_dir: str = "result",
    topk_tokens: int = 10,
    topk_subspaces: int = 50,
    weight_type: str = "c_proj",
    interp_type: str = "detector",
    with_negative: bool = True,
    save_file: bool = True,
    return_heatmap: bool = True,
    device: str | None = None
):
  
    os.makedirs(out_dir, exist_ok=True)
    layers_to_use = parse_layers_arg(layers, model_name)
    print(f">> Layers to process: {layers_to_use}")
    
    interp_io = MLP_DEEF_INTERP(model_name=model_name, output_dir=out_dir, device=device)
 
    # Run analysis for each layer
    with torch.no_grad():
        for layer_idx in layers_to_use:
            print(f"\n[INFO] Processing layer {layer_idx} ...")
            interp_io.mlp_subspace_interp(
                layer_idx=layer_idx,
                out_dir=out_dir,
                topk_tokens=topk_tokens,
                topk_subspaces=topk_subspaces,
                weight_type=weight_type,
                interp_type=interp_type,
                with_negative=with_negative,
                save_file=save_file,
                return_heatmap=return_heatmap
            )

    print("\nAll specified layers have been processed.")

 

if __name__ == "__main__":
    args = parse_args()
    
    run_mlp_analysis(
        model_name=args.model_name,
        layers=args.layers,
        out_dir=args.out_dir,
        topk_tokens=args.topk_tokens,
        topk_subspaces=args.topk_subspaces,
        weight_type=args.weight_type,
        interp_type=args.interp_type,
        with_negative=args.with_negative,
        save_file=args.save_file,
        return_heatmap=args.return_heatmap,
        device=args.device
    )
 
