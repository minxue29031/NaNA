import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import argparse
from typing import List
from ke.edit_compare import MLPEditor
from block_interp.model_load import parse_layers_arg
from ke.eval_gene import evaluate_generalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Knowledge Editing Pipeline (KE)")

    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Hugging Face model name")
    parser.add_argument("--input_text", type=str, default="The cat looks very", help="Input context used for triggering the edited behavior")
    parser.add_argument("--original_target", type=str, default=" happy", help="Original next-token to suppress")
    parser.add_argument("--new_target", type=str, default=" cute", help="New next-token to boost after editing")
    parser.add_argument("--layers", nargs='+', default=["all"], help="Specify layers (e.g. --layers 4 5 6) or 'all'")
    parser.add_argument("--weight_type", type=str, default="c_proj", choices=["c_proj", "c_fc"], help="Which MLP matrix to modify")
    parser.add_argument("--delta_new_boost", type=float, default=0.8, help="Boost strength for new target")
    parser.add_argument("--delta_new_suppress", type=float, default=0.8, help="Suppression strength for new target")
    parser.add_argument("--delta_ori_suppress", type=float, default=0.8, help="Suppression strength for original target")
    parser.add_argument("--interp_type", type=str, default="all", choices=["all", "pos", "neg"], help="Interpretation subspace type")
    parser.add_argument("--circuit_mode", type=str, default="DeEf", help="Circuit mode (DeEf / Ef / De)")
    parser.add_argument("--edit_subspaces", type=int, default=15, help="How many top singular directions to apply editing to")
    parser.add_argument("--out_dir", type=str, default="result/ke", help="Directory to save KE logs / weights")

    return parser.parse_args()


#   RUN MLP EDITING
def run_editing_pipeline(
    model_name: str,
    input_text: str,
    layers: str,
    original_target: str,
    new_target: str,
    weight_type: str,
    delta_new_boost: float,
    delta_new_suppress: float,
    delta_ori_suppress: float,
    interp_type: str,
    circuit_mode: str,
    topk_subspaces: int,
    out_dir: str,
):

    os.makedirs(out_dir, exist_ok=True)
    layers_to_use = parse_layers_arg(layers, model_name)
    print(f">> Layers to edit: {layers_to_use}")

    # Load editor
    editor = MLPEditor(model_name=model_name)
    edited_model, original_model = editor.run_full_pipeline(
        input_text=input_text,
        layers_to_edit=layers_to_use,
        original_target=original_target,
        new_target=new_target,
        weight_type=weight_type,
        delta_new_boost=delta_new_boost,
        delta_new_suppress=delta_new_suppress,
        delta_ori_suppress=delta_ori_suppress,
        interp_type=interp_type,
        circuit_mode=circuit_mode,
        topk_subspaces=topk_subspaces,
        output_dir=out_dir,
    )

    print("\n[Done] Knowledge Editing pipeline completed.")
    print(f"Results saved at: {out_dir}")

 #   evaluate_generalization(original_model, edited_model, editor.tokenizer, csv_path=os.path.join(out_dir, "generalization_eval.csv"))


 
if __name__ == "__main__":
    args = parse_args()

    run_editing_pipeline(
        model_name=args.model_name,
        input_text=args.input_text,
        layers=args.layers,
        original_target=args.original_target,
        new_target=args.new_target,
        weight_type=args.weight_type,
        delta_new_boost=args.delta_new_boost,
        delta_new_suppress=args.delta_new_suppress,
        delta_ori_suppress=args.delta_ori_suppress,
        interp_type=args.interp_type,
        circuit_mode=args.circuit_mode,
        topk_subspaces=args.edit_subspaces,
        out_dir=args.out_dir,
    )
