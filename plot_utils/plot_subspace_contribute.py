import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

def plot_subspace(
    data,
    output_file="mlp_contrib.png",
    cmap='bwr',
    marker='o',
    size_scale=200,
    topk_subspaces=None,
    color_threshold=None,
):
    """
    Plot MLP subspace contributions from a dictionary.
    """
    layers, subspaces, contributions = [], [], []

    for layer_name, layer_data in data.items():
        layer_idx = layer_data['layer_idx']
        subspace_results = layer_data['subspace_results']

        if topk_subspaces is not None:
            subspace_results = sorted(
                subspace_results, key=lambda x: abs(x['contribution']), reverse=True
            )[:topk_subspaces]

        for subspace in subspace_results:
            layers.append(layer_idx)
            subspaces.append(subspace['subspace_index'])
            contributions.append(subspace['contribution'])

    values = np.array(contributions)

    # Apply color threshold if specified
    if color_threshold is not None:
        values_for_color = np.clip(values, -color_threshold, color_threshold)
        vmin, vmax = -color_threshold, color_threshold
    else:
        values_for_color = values
        max_abs = np.max(np.abs(values_for_color)) + 1e-8
        vmin, vmax = -max_abs, max_abs

    # Linear color mapping
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        layers,
        subspaces,
        s=[abs(c) * size_scale for c in contributions],
        c=values_for_color,
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        marker=marker,
        edgecolors="k",
    )

    plt.xlabel("Layer")
    plt.ylabel("Subspace Index")
    plt.title("MLP Subspace Contribution (Linear)")
    plt.xticks(sorted(set(layers)))
    plt.grid(True)


    cbar = plt.colorbar(sc, label=f"Contribution (linear, threshold applied)")
    legend_element = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label="Contribution",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="k",
        )
    ]
    plt.legend(handles=legend_element, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" Saved subspace contribution plot to {output_file}")
