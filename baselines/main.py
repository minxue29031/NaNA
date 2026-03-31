import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
from baselines import config
from baselines.models import load_resources
from baselines.utils import load_dataset, load_baseline_json
from baselines.evaluator import cache_activations, Evaluator, run_loop

 
def plot_results(
    eval_steps, 
    tc_data, 
    sae_vanilla_data, 
    sae_jumprelu_data, 
    sae_topk_data, 
    sae_batchtopk_data,
    mlp_data, 
    full_mlp, 
    full_tc, 
    rand_data, 
    topk_data, 
    metric_name="Probability"
):

    """Generic plotting function for both Probability and Rank."""
    rand_x, rand_vals = rand_data
    topk_x, topk_vals = topk_data

    sca_map = dict(zip(topk_x, topk_vals))
    sca_plot_x = []
    sca_plot_y = []
    
    for step in eval_steps:
        if step in sca_map:
            sca_plot_x.append(step)
            sca_plot_y.append(sca_map[step])
            
    rand_map = dict(zip(rand_x, rand_vals))
    rand_plot_x = []
    rand_plot_y = []
    for step in eval_steps:
        if step in rand_map:
            rand_plot_x.append(step)
            rand_plot_y.append(rand_map[step])
 
    plt.figure(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
    sns.set_theme(style="ticks")
    ax = plt.gca()
    ax.grid(False)
 
    if len(sca_plot_x) > 0:
        plt.plot(sca_plot_x, sca_plot_y, label='Top-K SCA', 
                 linewidth=config.LINE_WIDTH, color='tab:red', 
                 markersize=config.MARKER_SIZE)  
                 
   # if len(rand_plot_x) > 0:
   #     plt.plot(rand_plot_x, rand_plot_y, label='Random DEU', 
   #              linewidth=config.LINE_WIDTH, color='purple', linestyle=':')
 
 

    # Plot Curves
    plt.plot(eval_steps, tc_data, label='TransCoder', linewidth=config.LINE_WIDTH, color='tab:blue', marker='o', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, sae_vanilla_data, label='SAE Vanilla', linewidth=config.LINE_WIDTH, color='tab:orange', linestyle='-.', marker='s', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, sae_jumprelu_data, label='SAE JumpReLU', linewidth=config.LINE_WIDTH, color='tab:pink', linestyle='-.', marker='D', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, sae_topk_data, label='SAE TopK', linewidth=config.LINE_WIDTH, color='tab:purple', linestyle='-.', marker='^', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, sae_batchtopk_data, label='SAE BatchTopK', linewidth=config.LINE_WIDTH, color='tab:cyan', linestyle='-.', marker='v', markersize=config.MARKER_SIZE)
    plt.plot(eval_steps, mlp_data, label='MLP', linewidth=config.LINE_WIDTH, color='tab:green', marker='^', markersize=config.MARKER_SIZE)


    # Plot Full Model Baselines (Horizontal Lines)
    plt.axhline(y=full_mlp, color='tab:green', linestyle='--', linewidth=config.LINE_WIDTH, label='Full MLP')
    plt.axhline(y=full_tc, color='tab:blue', linestyle='--', linewidth=config.LINE_WIDTH, label='Full TC')
    
    plt.xlim(-5, 146) 
    plt.xticks(range(0, 141, 20))  
    
    plt.ylabel(f'Average Target {metric_name}', fontsize=config.LABEL_SIZE)
    plt.xlabel('Number of Subspaces/Features ($k$)', fontsize=config.LABEL_SIZE)
    plt.title(f'Layer {config.LAYER_IDX} - {metric_name} Recovery', fontsize=config.TITLE_SIZE)
    plt.xticks(fontsize=config.TICK_SIZE)
    plt.yticks(fontsize=config.TICK_SIZE)
    plt.legend(loc='upper right', frameon=False, fontsize=config.LEGEND_SIZE, labelspacing=0.05)
 
 
    plt.tight_layout()
    save_path = os.path.join(config.SAVE_DIR, f'target_{metric_name.lower()}_layer_{config.LAYER_IDX}_{config.DATASET_NAME}.png')
    plt.savefig(save_path, dpi=300)
    print(f"{metric_name} Plot saved to {save_path}")
    plt.show()




def main():

    model, tokenizer, transcoder, sae_vanilla, sae_jumprelu, sae_topk, sae_batchtopk = load_resources()
    prompts, target_ids = load_dataset(config.DATA_PATH, model)
    rand_x, rand_probs, rand_ranks = load_baseline_json(config.RANDOM_PROB_PATH, config.RANDOM_RANK_PATH)
    topk_x, topk_probs, topk_ranks = load_baseline_json(config.TOPK_GEN_PROB_PATH, config.TOPK_GEN_RANK_PATH)

    # Cache Activations and Initialize Evaluator 
    transcoder_activs, mlp_activs, sae_vanilla_activs, sae_jumprelu_activs, \
    sae_topk_activs, sae_batchtopk_activs = cache_activations(
        model,
        transcoder,
        sae_vanilla,
        sae_jumprelu,
        sae_topk,
        sae_batchtopk,
        prompts
    )
    

    # 初始化 Evaluator
    evaluator_vanilla = Evaluator(model, transcoder, sae_vanilla, prompts, target_ids)
    evaluator_jumprelu = Evaluator(model, transcoder, sae_jumprelu, prompts, target_ids)
    evaluator_topk = Evaluator(model, transcoder, sae_topk, prompts, target_ids)
    evaluator_batchtopk = Evaluator(model, transcoder, sae_batchtopk, prompts, target_ids)

      
    # Calculate Full Baselines
    print("Calculating Full Baselines...")
    full_mlp_prob, full_mlp_rank = evaluator_vanilla.eval_full_original()
    full_tc_prob, full_tc_rank = evaluator_vanilla.eval_full_transcoder()

    # Run Top-K Curves
    print(f"Calculating Top-k Features Curves on steps: {config.EVAL_STEPS}")
    mlp_probs, mlp_ranks = run_loop(evaluator_vanilla.eval_mlp_on_num, mlp_activs, config.EVAL_STEPS, "MLP Top-k")
    tc_probs, tc_ranks = run_loop(evaluator_vanilla.eval_tc_on_num, transcoder_activs, config.EVAL_STEPS, "TC Top-k")

    sae_vanilla_probs, sae_vanilla_ranks = run_loop(evaluator_vanilla.eval_sae_on_num, sae_vanilla_activs, config.EVAL_STEPS, "SAE Vanilla Top-k")
    sae_jumprelu_probs, sae_jumprelu_ranks = run_loop(evaluator_jumprelu.eval_sae_on_num, sae_jumprelu_activs, config.EVAL_STEPS, "SAE JumpReLU Top-k")
    sae_topk_probs, sae_topk_ranks = run_loop(evaluator_topk.eval_sae_on_num, sae_topk_activs, config.EVAL_STEPS, "SAE TopK Top-k")
    sae_batchtopk_probs, sae_batchtopk_ranks = run_loop(evaluator_batchtopk.eval_sae_on_num, sae_batchtopk_activs, config.EVAL_STEPS, "SAE BatchTopK Top-k")
        
    

    # Plotting
    print("Plotting...")
    plot_results(
        config.EVAL_STEPS, 
        tc_probs, 
        sae_vanilla_probs, 
        sae_jumprelu_probs, 
        sae_topk_probs, 
        sae_batchtopk_probs,
        mlp_probs, 
        full_mlp_prob, 
        full_tc_prob, 
        (rand_x, rand_probs), 
        (topk_x, topk_probs),
        metric_name="Probability"
    )
    
    plot_results(
        config.EVAL_STEPS, 
        tc_ranks, 
        sae_vanilla_ranks, 
        sae_jumprelu_ranks, 
        sae_topk_ranks, 
        sae_batchtopk_ranks,
        mlp_ranks, 
        full_mlp_rank, 
        full_tc_rank, 
        (rand_x, rand_ranks), 
        (topk_x, topk_ranks),
        metric_name="Rank"
    )

if __name__ == "__main__":
    main()