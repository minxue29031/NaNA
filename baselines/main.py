import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
from baselines import config
from baselines.models import load_resources
from baselines.utils import load_dataset, load_baseline_json
from baselines.evaluator import cache_activations, Evaluator, run_loop

def plot_results(
    eval_steps, 
    tc_data, sae_data, mlp_data, 
    full_mlp, full_tc, 
    rand_data, topk_data, 
    metric_name="Probability"
):
    """Generic plotting function for both Probability and Rank."""
    tc_vals, sae_vals, mlp_vals = tc_data, sae_data, mlp_data
    rand_x, rand_vals = rand_data
    topk_x, topk_vals = topk_data
    
    plt.figure(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
    sns.set_theme(style="white")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Plot External Baselines
    if len(rand_x) > 0:
        plt.plot(rand_x, rand_vals, label='Random SCA', linewidth=config.LINE_WIDTH, color='purple', linestyle=':')
    if len(topk_x) > 0:
        plt.plot(topk_x, topk_vals, label='Top-K SCA', linewidth=config.LINE_WIDTH, color='tab:red')

    # Plot Curves
    plt.plot(eval_steps, tc_vals, label='Top-k TC', linewidth=config.LINE_WIDTH, color='tab:blue', marker='o', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, sae_vals, label='Top-k SAE', linewidth=config.LINE_WIDTH, color='tab:orange', linestyle='-.', marker='s', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, mlp_vals, label='Top-k MLP', linewidth=config.LINE_WIDTH, color='tab:green', marker='^', markersize=config.MARKER_SIZE)

    # Plot Full Model Baselines (Horizontal Lines)
    plt.axhline(y=full_mlp, color='tab:green', linestyle='--', linewidth=config.LINE_WIDTH, label='Full MLP')
    plt.axhline(y=full_tc, color='tab:blue', linestyle='--', linewidth=config.LINE_WIDTH, label='Full TC')

    plt.ylabel(f'Average Target {metric_name}', fontsize=config.LABEL_SIZE)
    plt.xlabel('Number of features/neurons used', fontsize=config.LABEL_SIZE)
    plt.title(f'Layer {config.LAYER_IDX} - {metric_name} Recovery', fontsize=config.TITLE_SIZE)
    plt.xticks(fontsize=config.TICK_SIZE)
    plt.yticks(fontsize=config.TICK_SIZE)
    plt.legend(loc='upper right', frameon=False, fontsize=config.LEGEND_SIZE, labelspacing=0.1)

    plt.tight_layout()
    save_path = os.path.join(config.SAVE_DIR, f'target_{metric_name.lower()}_layer_{config.LAYER_IDX}.png')
    plt.savefig(save_path, dpi=300)
    print(f"{metric_name} Plot saved to {save_path}")
    plt.show()

def main():
    model, tokenizer, transcoder, sae = load_resources()
    prompts, target_ids = load_dataset(config.DATA_PATH, model)
    rand_x, rand_probs, rand_ranks = load_baseline_json(config.RANDOM_PROB_PATH, config.RANDOM_RANK_PATH)
    topk_x, topk_probs, topk_ranks = load_baseline_json(config.TOPK_GEN_PROB_PATH, config.TOPK_GEN_RANK_PATH)

    # Cache Activations and Initialize Evaluator
    transcoder_activs, mlp_activs, sae_activs = cache_activations(model, transcoder, sae, prompts)
    evaluator = Evaluator(model, transcoder, sae, prompts, target_ids)
    
    # Calculate Full Baselines
    print("Calculating Full Baselines...")
    full_mlp_prob, full_mlp_rank = evaluator.eval_full_original()
    full_tc_prob, full_tc_rank = evaluator.eval_full_transcoder()

    # Run Top-K Curves
    print(f"Calculating Top-k Features Curves on steps: {config.EVAL_STEPS}")
    mlp_probs, mlp_ranks = run_loop(evaluator.eval_mlp_on_num, mlp_activs, config.EVAL_STEPS, "MLP Top-k")
    tc_probs, tc_ranks = run_loop(evaluator.eval_tc_on_num, transcoder_activs, config.EVAL_STEPS, "TC Top-k")
    sae_probs, sae_ranks = run_loop(evaluator.eval_sae_on_num, sae_activs, config.EVAL_STEPS, "SAE Top-k")

    # Plotting
    print("Plotting...")
    plot_results(
        config.EVAL_STEPS, 
        tc_probs, sae_probs, mlp_probs, 
        full_mlp_prob, full_tc_prob, 
        (rand_x, rand_probs), (topk_x, topk_probs),
        metric_name="Probability"
    )
    
    plot_results(
        config.EVAL_STEPS, 
        tc_ranks, sae_ranks, mlp_ranks, 
        full_mlp_rank, full_tc_rank, 
        (rand_x, rand_ranks), (topk_x, topk_ranks),
        metric_name="Rank"
    )

if __name__ == "__main__":
    main()