import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.colors as mcolors

def plot_subspace_flow(
    data,
    output_file="mlp_flow.png",
    color_label="Contribution",
    title="MLP Subspace Contribution Flow",
    cmap="bwr",
    marker="o",
    legend_label="Subspace",
    color_threshold=None,
    box_width=0.8,
    left_margin=0.08,
    right_margin=0.92,
    size_scale=400,
    min_size=20   
):
    """
    Plot an MLP subspace contribution flow diagram.
    """

    layers, subspaces, contributions = [], [], []
    for layer_name, layer_data in data.items():
        layer_idx = layer_data["layer_idx"]
        for subspace in layer_data["subspace_results"]:
            layers.append(layer_idx)
            subspaces.append(subspace["subspace_index"])
            contributions.append(subspace["contribution"])

    layers = np.array(layers)
    subspaces = np.array(subspaces)
    contributions = np.array(contributions)

    if color_threshold is not None:
        color_vals = np.clip(contributions, -color_threshold, color_threshold)
        vmin, vmax = -color_threshold, color_threshold
    else:
        color_vals = contributions
        max_abs = np.max(np.abs(color_vals)) + 1e-8
        vmin, vmax = -max_abs, max_abs

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    abs_vals = np.abs(contributions)
    sizes = abs_vals * size_scale
    sizes = np.clip(sizes, min_size, None) 

    layer_counts = {}
    y_ordered = []
    for layer in layers:
        count = layer_counts.get(layer, 0)
        y_ordered.append(count)
        layer_counts[layer] = count + 1

    fig, ax = plt.subplots(figsize=(18, 6))

    sc = ax.scatter(
        layers, y_ordered,
        s=sizes,
        c=color_vals,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        marker=marker,
        edgecolors="#333333",
        linewidths=1.2
    )

    unique_layers = sorted(set(layers))
    layer_y_range = {}
    for layer in unique_layers:
        ys = [y for x, y in zip(layers, y_ordered) if x == layer]
        layer_y_range[layer] = (min(ys)-0.3, max(ys)+0.3)
        rect = Rectangle(
            (layer-box_width/2, layer_y_range[layer][0]),
            box_width,
            layer_y_range[layer][1]-layer_y_range[layer][0],
            fill=False, edgecolor="gray", linewidth=1.5
        )
        ax.add_patch(rect)

    for i in range(len(unique_layers)-1):
        x_start = unique_layers[i]+box_width/2
        x_end = unique_layers[i+1]-box_width/2
        y_mid_start = np.mean(layer_y_range[unique_layers[i]])
        y_mid_end = np.mean(layer_y_range[unique_layers[i+1]])
        ax.annotate('', xy=(x_end, y_mid_end), xytext=(x_start, y_mid_start),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    for xi, yi, label in zip(layers, y_ordered, subspaces):
        ax.text(xi, yi, str(label), fontsize=8, ha="center", va="center")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Subspace (Per-Layer Order)")
    ax.set_title(title)
    ax.set_xticks(unique_layers)
    ax.set_yticks(range(max(y_ordered)+1))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(color_label)

    legend_element = [
        Line2D([0], [0], marker=marker, color="w", label=legend_label,
               markerfacecolor="gray", markersize=10, markeredgecolor="k")
    ]
    ax.legend(handles=legend_element, loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=1)

    fig.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" Saved subspace contribution flow plot to {output_file}")
