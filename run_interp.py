import os
import torch
import argparse
from typing import List
from block_interp.interp_mlp import MLP_DEEF_INTERP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Professional MLP Subspace Interpretation Tool"
    )
    parser.add_argument("--model_name", type=str, default="gpt2-medium",
                        help="Hugging Face model name (e.g., 'gpt2-medium')")
    parser.add_argument("--layers", type=int, nargs="+", default=[16],
                        help="Layer indices to analyze (space-separated)")
    parser.add_argument("--out_dir", type=str, default="result",
                        help="Directory to save results")
    parser.add_argument("--topk_tokens", type=int, default=10,
                        help="Number of top tokens per direction")
    parser.add_argument("--topk_subspaces", type=int, default=50,
                        help="Number of top singular directions to analyze")
    parser.add_argument("--weight_type", type=str, choices=["c_proj", "c_fc", "ov"],
                        default="c_proj", help="Weight matrix type")
    parser.add_argument("--interp_type", type=str, choices=["detector", "effector"],
                        default="detector", help="Interpretation type")
    parser.add_argument("--with_negative", action="store_true",
                        help="Save negative directions as well")
    parser.add_argument("--use_activation", action="store_true",
                        help="Apply activation function in projection")
    parser.add_argument("--with_values", action="store_true",
                        help="Include token scores in output")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cuda/cpu); auto-detect if None")
    return parser.parse_args()

def run_mlp_analysis(
    model_name: str,
    layers_to_use: List[int],
    out_dir: str = "result",
    topk_tokens: int = 10,
    topk_subspaces: int = 50,
    weight_type: str = "c_proj",
    interp_type: str = "detector",
    with_negative: bool = True,
    use_activation: bool = False,
    with_values: bool = False,
    device: str | None = None
):
 
 
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

 
    os.makedirs(out_dir, exist_ok=True)

    interp_io = MLP_DEEF_INTERP(
        model_name=model_name,
        output_dir=out_dir,
        device=device
    )

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
            use_activation=use_activation,
            with_values=with_values
        )

    print("\n[INFO] ✅ All specified layers have been processed.")

 

if __name__ == "__main__":
    args = parse_args()
    run_mlp_analysis(
        model_name=args.model_name,
        layers_to_use=args.layers,
        out_dir=args.out_dir,
        topk_tokens=args.topk_tokens,
        topk_subspaces=args.topk_subspaces,
        weight_type=args.weight_type,
        interp_type=args.interp_type,
        with_negative=args.with_negative,
        use_activation=args.use_activation,
        with_values=args.with_values,
        device=args.device
    )
