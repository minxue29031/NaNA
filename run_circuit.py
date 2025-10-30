import os
import torch
import argparse
from block_interp.interp_mlp import load_model_and_embeddings
from circuit.circuit_analysis import analyze_mlp_subspaces
from circuit.collect_circuit_info import save_circuit_info, layer_info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--return_heatmap", action="store_true", help="Return heatmap of each direction")
    parser.add_argument("--size_scale", type=float, default=200.0, help="Size scale for circuit plot")
    parser.add_argument("--color_threshold", type=float, default=2.0, help="Color threshold for circuit plot")
    parser.add_argument("--box_width", type=float, default=0.7, help="Box width for circuit plot")

    return parser.parse_args()


def extract_circuit(
    model_name="gpt2-medium",
    in_seq="The cat looks very",
    target_word=" happy",
    layers="16",
    topk_subspaces=15,
    topk_tokens=20,
    output_dir="result/circuit",
    circuit_mode="DeEf",
    interp_type="all",
    weight_type="c_fc",
    return_heatmap=False,
    size_scale=200.0,
    color_threshold=2.0,
    box_width=0.7
):

    os.makedirs(output_dir, exist_ok=True)


    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)

    if isinstance(layers, str) and layers.lower() == "all":
        total_layers = len([m for m in model.transformer.h if hasattr(m.mlp, "c_fc")])
        layers_to_use = list(range(total_layers))
    else:
        layers_to_use = [int(x) for x in str(layers).replace(",", " ").split()]

    all_layers_circuits = {}
    all_cirpoints_scores = {}

    for layer_idx in layers_to_use:
        print(f"\n===== Processing Layer {layer_idx} =====")

        reshape_W_E, U, S, Vh, layer_io, _, _, _, _, _, _ = layer_info(
            model, tokenizer, W_E, in_seq, layer_idx, device, weight_type
        )

        layer_circuit, circuit_point_score = analyze_mlp_subspaces(
            model_name,
            tokenizer,
            W_E,
            reshape_W_E,
            U,
            S,
            Vh,
            layer_idx,
            layer_io,
            target_word,
            topk_subspaces,
            weight_type,
            circuit_mode,
            interp_type,
            output_dir,
            device,
            topk_tokens,
            return_heatmap
        )

        all_layers_circuits[f"layer_{layer_idx}"] = layer_circuit
        all_cirpoints_scores[f"layer_{layer_idx}"] = circuit_point_score

    # Save results
    save_circuit_info(
        output_dir,
        model_name,
        weight_type,
        circuit_mode,
        all_layers_circuits,
        all_cirpoints_scores,
        size_scale,
        color_threshold,
        box_width
    )

    print("\n=== All Executions Completed Successfully ===")


if __name__ == "__main__":
    args = parse_args()

    extract_circuit(
        model_name=args.model_name,
        in_seq=args.in_seq,
        target_word=args.target_word,
        layers=args.layers,
        topk_subspaces=args.topk_subspaces,
        topk_tokens=args.topk_tokens,
        output_dir=args.output_dir,
        circuit_mode=args.circuit_mode,
        interp_type=args.interp_type,
        weight_type=args.weight_type,
        return_heatmap=args.return_heatmap,
        size_scale=args.size_scale,
        color_threshold=args.color_threshold,
        box_width=args.box_width
    )
