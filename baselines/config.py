import os

 
LAYER_IDX = 10
SAVE_DIR = "baseline_result"

# Dataset Paths
DATA_PATH = "dataset/baseline_prompt_targetori.json"
SAE_PATH = "dataset/sae/sae_weights.safetensors"
TRANSCODER_TEMPLATE = "./dataset/transcoder/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

# Baseline Result Paths
ROOT_BASE_DIR = "/nfs/data/projects/minxue/svd_directions/baseline_result"
RANDOM_DIR = os.path.join(ROOT_BASE_DIR, f"result_prob_rank_single_layer_{LAYER_IDX}/random_general/summary_jsons")
RANDOM_PROB_PATH = os.path.join(RANDOM_DIR, "random_general_avg_prob_vs_topN.json")
RANDOM_RANK_PATH = os.path.join(RANDOM_DIR, "random_general_avg_rank_vs_topN.json")

TOPK_GEN_DIR = os.path.join(ROOT_BASE_DIR, f"result_prob_rank_single_layer_{LAYER_IDX}/top_k_general/summary_jsons")
TOPK_GEN_PROB_PATH = os.path.join(TOPK_GEN_DIR, "top_k_general_avg_prob_vs_topN.json")
TOPK_GEN_RANK_PATH = os.path.join(TOPK_GEN_DIR, "top_k_general_avg_rank_vs_topN.json")

# Evaluation Steps
EVAL_STEPS = [
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150
]


# Visual Config (Plotting)
FIG_WIDTH = 5
FIG_HEIGHT = 3.5
TITLE_SIZE = 12
LABEL_SIZE = 12
TICK_SIZE = 12
LEGEND_SIZE = 8
LINE_WIDTH = 1
MARKER_SIZE = 1.5

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)