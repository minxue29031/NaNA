import os
import torch
import json
from plot_utils.plot_subspace_contribute import plot_subspace
from plot_utils.plot_path import plot_subspace_flow


def save_circuit_info(
    output_dir, 
    model_name, 
    weight_type, 
    circuit_mode, 
    all_layers_circuits, 
    all_cirpoints_scores,  
    size_scale=100, 
    color_threshold=10, 
    box_width=0.7
):
    """
    Save circuit analysis results and circuit point scores to JSON files,
    and generate subspace flow and contribution plots under structured folder.
    """
    save_dir = os.path.join(
        output_dir, f"{model_name}_circuit", f"MLP_{weight_type}", circuit_mode
    )
    os.makedirs(save_dir, exist_ok=True)

    circuits_path = os.path.join(save_dir, f"circuit_{weight_type}_{model_name}.json")
    cirpoints_scores_path = os.path.join(
        save_dir, f"circuit_points_scores_{weight_type}_{model_name}.json"
    )

    with open(circuits_path, "w", encoding="utf-8") as f:
        json.dump(all_layers_circuits, f, indent=4, ensure_ascii=False)

    with open(cirpoints_scores_path, "w", encoding="utf-8") as f:
        json.dump(all_cirpoints_scores, f, indent=4, ensure_ascii=False)

    # Plot subspace flow among layers
    flow_image_path = os.path.join(save_dir, "circuit.png")
    plot_subspace_flow(
        data=all_cirpoints_scores,
        output_file=flow_image_path,
        color_threshold=color_threshold,
        size_scale=size_scale,
        box_width=box_width
    )
    
    # Plot subspace contribution
    subspace_image_path = os.path.join(save_dir, "subspace_contribution.png")
    plot_subspace(
        data=all_cirpoints_scores,
        output_file=subspace_image_path,
        size_scale=size_scale,
        color_threshold = color_threshold
    )

    print(f"\n Saved all files to {save_dir}")