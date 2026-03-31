import os

 
# Dataset Paths
LAYER_IDX = 10
DATASET_NAME = "simple_SVA"  #within_rc_SVA   simple_SVA   gender_pron  baseline_prompt_target_gpt2 
DATA_PATH = f"dataset/{DATASET_NAME}.json"   

SAVE_DIR = "REBUTTAL_baseline_result/gpt2/result_baselines"
SAE_VANILLA_PATH = "dataset/vanilla_sae/sae_weights.safetensors"
SAE_TOPK_PATH = "dataset/topk_sae/sae.pt"
SAE_BATCHTOPK_PATH = "dataset/batchtopk_sae/sae.pt"
SAE_JUMPRELU_PATH = "dataset/jumprelu_sae/sae.pt"
TRANSCODER_PATH = "dataset/transcoder/final_sparse_autoencoder_gpt2-small_blocks.10.ln2.hook_normalized_24576.pt"


# Baseline Result Paths
ROOT_BASE_DIR = "REBUTTAL_baseline_result/gpt2/summary_jsons"
RANDOM_PROB_PATH = os.path.join(ROOT_BASE_DIR, f"stats_result_{DATASET_NAME}_gpt2_general_avg_prob.json")
RANDOM_RANK_PATH = os.path.join(ROOT_BASE_DIR, f"stats_result_{DATASET_NAME}_gpt2_general_avg_rank.json")

TOPK_GEN_PROB_PATH = os.path.join(ROOT_BASE_DIR, f"stats_result_{DATASET_NAME}_gpt2_general_avg_prob.json")
TOPK_GEN_RANK_PATH = os.path.join(ROOT_BASE_DIR, f"stats_result_{DATASET_NAME}_gpt2_general_avg_rank.json")
 
 

"""
# Baseline Result Paths
ROOT_BASE_DIR = "/nfs/data/projects/minxue/svd_directions/baseline_result/gpt2"
RANDOM_DIR = os.path.join(ROOT_BASE_DIR, f"result_prob_rank_single_layer_{LAYER_IDX}/random_general/summary_jsons")
RANDOM_PROB_PATH = os.path.join(RANDOM_DIR, "random_general_avg_prob_vs_topN.json")
RANDOM_RANK_PATH = os.path.join(RANDOM_DIR, "random_general_avg_rank_vs_topN.json")

TOPK_GEN_DIR = os.path.join(ROOT_BASE_DIR, f"result_prob_rank_single_layer_{LAYER_IDX}/top_k_general/summary_jsons")
TOPK_GEN_PROB_PATH = os.path.join(TOPK_GEN_DIR, "top_k_general_avg_prob_vs_topN.json")
TOPK_GEN_RANK_PATH = os.path.join(TOPK_GEN_DIR, "top_k_general_avg_rank_vs_topN.json")
"""

# Evaluation Steps
EVAL_STEPS = [
1, 5, 10, 15, 20, 25, 30, 35,
40, 45, 50, 55, 60, 65, 70, 75, 
80, 85, 90, 95, 100, 105, 110, 
115, 120, 125, 130, 135, 140
]


# Visual Config (Plotting)
FIG_WIDTH = 4.3
FIG_HEIGHT = 3.2
TITLE_SIZE = 11
LABEL_SIZE = 11
TICK_SIZE = 11
LEGEND_SIZE = 9
LINE_WIDTH = 1
MARKER_SIZE = 1.5

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)