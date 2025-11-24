import os
import matplotlib.pyplot as plt
import numpy as np


def safe_label(s):
    if s is None:
        return ""
    return str(s).replace("$", r"\$")   

def plot_subspace_heatmap(
    results, model_name, layer_idx, weight_type, interp_type, sign,
    out_dir, cell_width=1.0, cell_height=0.2, fontsize=9
    ):

    save_dir = os.path.join(
        out_dir, f"{model_name}_interp_dir", 
        "heatmap", f"MLP_{weight_type}_layer{layer_idx}"
    )
    os.makedirs(save_dir, exist_ok=True)


    directions = len(results)
    tokens_per_dir = max(len(r["top_tokens"]) for r in results)

    heatmap = np.zeros((tokens_per_dir, directions))
    token_text = np.empty((tokens_per_dir, directions), dtype=object)

    # Fill the matrix and normalize each column
    for col, r in enumerate(results):
        col_values = []
        for row, t in enumerate(r["top_tokens"]):
            if row < tokens_per_dir:
                value = t["value"] if t["value"] is not None else 0.0
                col_values.append(value)
                token_text[row, col] = t["token"]

        col_values = np.array(col_values)
        col_min, col_max = col_values.min(), col_values.max()

        if sign == "positive":
            # Normalize positive values to [0, 1] 
            norm_values = (col_values - col_min) / (col_max - col_min + 1e-8)
        else:
            # Normalize negative values to [0, -1]
            norm_values = -(col_values - col_min) / (col_max - col_min + 1e-8)

        for row, v in enumerate(norm_values):
            heatmap[row, col] = v

    # Set cells without a token to np.nan
    for i in range(tokens_per_dir):
        for j in range(directions):
            if token_text[i, j] is None:
                heatmap[i, j] = np.nan

    cmap = "Reds" if sign == "positive" else "Blues"
    fig_width = directions * cell_width
    fig_height = tokens_per_dir * cell_height
    plt.figure(figsize=(fig_width, fig_height))

    masked_heatmap = np.ma.masked_invalid(heatmap)

    im = plt.imshow(
        masked_heatmap, 
        aspect='auto', 
        cmap=cmap,
        vmin=-1.0 if sign=="negative" else 0,
        vmax=1.0 if sign=="positive" else 0
    )

    # Handle NaN display
    im.set_cmap(cmap)
    im.set_array(masked_heatmap)
    im.cmap.set_bad(color='white')  

    # Colorbar showing positive or negative normalized values
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label("Normalized token score")
    if sign == "negative":
        cbar.set_ticks([-1.0, -0.75, -0.5, -0.25, 0.0])
        cbar.set_ticklabels(["-1", "-0.75", "-0.5", "-0.25", "0"])
    else:
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0","0.25","0.5","0.75","1"])

    # Display tokens inside each cell
    for i in range(tokens_per_dir):
        for j in range(directions):
            token = token_text[i, j]
            if token is not None:
                val = heatmap[i, j]
                if sign == "positive":
                    color = 'white' if val > 0.5 else 'black'
                else:
                    color = 'black' if val < -0.5 else 'white'
                plt.text(j, i, safe_label(token), ha='center', va='center', 
                    fontsize=fontsize, color=color)

    plt.title(f"Layer {layer_idx} {interp_type} ({sign})")
    plt.xlabel("Subspace Index")
    plt.ylabel("Top Token")
    plt.xticks(
        ticks=np.arange(directions), 
        labels=[str(r["direction"]) for r in results]
    )
    plt.yticks(
        ticks=np.arange(tokens_per_dir), 
        labels=[f"T{i+1}" for i in range(tokens_per_dir)]
    )

    save_path = os.path.join(save_dir, f"{interp_type}_{sign}_heatmap.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"<< Heatmap saved to {save_path} >>")
