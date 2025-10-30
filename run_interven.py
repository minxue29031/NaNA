import os
import torch
import argparse
from circuit import *
from block_interp.interp_mlp import load_model_and_embeddings
from circuit.circuit_interv.model_interv import subspace_interv
from circuit.collect_circuit_info import layer_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPT-2 subspace intervention experiment"
    )
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="HuggingFace GPT-2 model name")
    parser.add_argument("--in_seq", type=str, default="The cat looks very", help="Input text sequence")
    parser.add_argument("--target_word", type=str, default=" happy", help="Expected next word (for reference)")
    parser.add_argument("--layer_idx", type=int, default=16, help="Target transformer layer index")
    parser.add_argument("--weight_type", type=str, default="c_proj", choices=["c_fc", "c_proj"], help="MLP weight type")
    parser.add_argument("--interv_mode", type=str, default="enhance", choices=["enhance", "suppress"], help="Intervention mode")
    parser.add_argument("--interv_dir_indices", type=int, nargs="+", default=[0], help="SVD direction indices to intervene")
    parser.add_argument("--interv_scale", type=float, default=0.8, help="Scaling factor for intervention")
    parser.add_argument("--topk_tokens", type=int, default=10, help="Number of top predicted tokens to display")
    parser.add_argument("--output_dir", type=str, default="result_probe", help="Output directory for logs/results")
    return parser.parse_args()


def extract_circuit_intervention(
    model_name="gpt2-medium",
    in_seq="The cat looks very",
    target_word=" happy",
    layer_idx=16,
    weight_type="c_fc",
    interv_mode="enhance",
    interv_dir_indices=[11],
    interv_scale=6.0,
    topk_tokens=10,
    output_dir="result_probe",
    device=None,
):
 

    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)

    reshape_W_E, U, S, Vh, layer_io, _, _, c_fc, c_proj, ln_2, act = layer_info(
        model,
        tokenizer,
        W_E,
        in_seq,
        layer_idx,
        device,
        weight_type
    )

    # Apply subspace intervention
    subspace_interv(
        model=model,
        tokenizer=tokenizer,
        U=U, S=S, Vh=Vh,
        layer_io=layer_io,
        layer_idx=layer_idx,
        W_E=W_E,
        reshape_W_E=reshape_W_E,
        weight_type=weight_type,
        interv_mode=interv_mode,
        interv_dir_indices=interv_dir_indices,
        interv_scale=interv_scale,
        return_toptoks=topk_tokens,
    )

    print("\n======== All Executions Completed Successfully =========")



if __name__ == "__main__":
    args = parse_args()

    extract_circuit_intervention(
        model_name=args.model_name,
        in_seq=args.in_seq,
        target_word=args.target_word,
        layer_idx=args.layer_idx,
        weight_type=args.weight_type,
        interv_mode=args.interv_mode,
        interv_dir_indices=args.interv_dir_indices,
        interv_scale=args.interv_scale,
        topk_tokens=args.topk_tokens,
        output_dir=args.output_dir,
        device=device,
    )
