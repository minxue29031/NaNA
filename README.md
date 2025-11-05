 
# 🧠 Transformers SVD Analysis — Detector & Effector

This repository provides tools for analyzing the semantic subspaces of MLP layers in transformer-based language models (e.g., "gpt2-medium"). By decomposing transformer blocks (e.g., MLPs) into sums of rank-1 subspaces: $W = \sum_{i=1}^{\text{rank}(W)} \sigma_i \  u_i \  v_i^T$, we can extract **detector** vectors ($u_i$) and **effector** vectors ($v_i^T$). These vectors reveal interpretable directions in the embedding space that correspond to meaningful linguistic or conceptual patterns.

## 🔍 Concept Overview

In Transformer MLP blocks,the **SVD decomposition** of these weight matrices reveals interpretable "directions" in hidden space:

* **Detector directions:** The input directions of the W_out subspace. Computes similarity with the embedding matrix to return the most highly correlated tokens.
* **Effector directions:** The output directions of the W_out subspace. Computes similarity with the embedding matrix to return the most highly correlated tokens.
  
By inspecting top tokens aligned with each singular direction, we can identify **semantic features** (e.g., sentiment, number, tense, named entities, etc.) captured by each MLP layer.



## 📂 Repository Structure

```
detector_effector/
├── run_interp.py              # Interpret MLP subspaces 
├── run_circuit.py             # Circuit extraction & analysis
├── run_modify.py              # Apply interventions 
├── block_interp/              # Core SVD & MLP processing modules
├── circuit/                   # Circuit analysis 
│   └── circuit_interv/        # Specific intervention implementations
├── plot_utils/                # Visualization utilities
├── result/                    
├── requirements.txt        
```
 


## 📦 Requirements

All dependencies are listed in `requirements.txt`.
To install:

```bash
pip install -r requirements.txt
```

## 🧩 Example Usage

### 🔹 MLP Subspace Interpretation

```bash
python run_interp.py \
    --model_name gpt2-medium \
    --layers 16 \
    --out_dir result \
    --topk_tokens 10 \
    --topk_subspaces 50 \
    --weight_type c_proj \
    --interp_type detector \
    --with_negative
```

* **Returns:**
  * Top tokens per subspace
  * heatmaps and datastore


### 🔹 Subspace Circuit Analysis

```bash
python run_circuit.py \
    --model_name gpt2-medium \
    --in_seq "The cat looks very" \
    --target_word " happy" \
    --layers 16 \
    --topk_subspaces 15 \
    --topk_tokens 20 \
    --output_dir result/circuit \
    --circuit_mode DeEf \
    --interp_type all \
    --weight_type c_fc \
    --return_heatmap
```
  * Circuit & Contribution scores
  * Top tokens per subspace
  * heatmaps and datastore
 
### 🔹 Subspace Intervention

Apply interventions to enhance or remove selected subspaces.

> **Note:** Run `run_circuit.py` first to generate `circuit_points_scores_{weight_type}_gpt2-medium.json` for analysis.

 
```bash
python run_modify.py \
    --gene_or_abla general \
    --weight_type c_fc \
    --top_subspaces 10 \
    --use_positive_only \
    --auto_subspace_file path/to/subspace_results.json \
    --model_name gpt2-medium \
    --layers 17 18 19 20 \
    --use_bias \
    --input_text "The cat looks very" \
    --output_dir result/interven_result \
    --modify_type rebuild \
    --interv_factor 0.1 \
    --use_full_residual \
    --token_num 20
```
 

### ⚙️ Configuration Parameters

Both scripts share similar configurable options:


| Argument               | Type     | Default         | Description                                                  |
| ---------------------- | -------- | --------------- | ------------------------------------------------------------ |
| `--model_name`         | str      | `"gpt2-medium"` | Model name. Options: `"gpt2"`, `"gpt2-medium"`, `"gpt2-xl"`  |
| `--layers`             | int list | `[16]`          | Layer indices to analyze                                     |
| `--out_dir`            | str      | `"result"`      | Directory to save results                                    |
| `--topk_tokens`        | int      | `10`            | Top-K tokens per singular direction                          |
| `--topk_subspaces`     | int      | `50`            | Number of top singular directions to analyze                 |
| `--weight_type`        | str      | `"c_proj"`      | MLP weight type: `c_proj`, `c_fc`                            |
| `--interp_type`        | str      | `"detector"`    | Interpretation type: `detector`, `effector`, or `all`        |
| `--with_negative`      | bool     | `False`         | Save negative directions as well                             |
| `--use_activation`     | bool     | `False`         | Apply activation function in projection                      |
| `--with_values`        | bool     | `False`         | Include token scores in output                               |
| `--gene_or_abla`       | str      | `"general"`     | Intervention mode: `"general"` or `"ablation"`               |
| `--use_positive_only`  | bool     | `False`         | Only include subspaces with positive contributions           |
| `--auto_subspace_file` | str      | Required        | Path to a JSON file containing top subspaces that were automatically extracted, enhanced, or suppressed |
| `--manual_subspace_file` | str      | Required      | Path to a JSON file containing top subspaces that were manually enhanced, or suppressed |
| `--interv_scale`       | float    | `0.8`           | Scaling factor for intervention effect                       |
| `--interv_dir_indices` | list     | `[6]`           | Subspace directions to intervene                             |
| `--return_toptoks`     | int      | `20`            | Number of top tokens to return after intervention            |
| `--use_bias`           | flag     | False           | Modify MLP using bias                                        |
| `--modify_type`        | str      | `"rebuild"`     | Type of subspace modification to apply: `"rebuild"`, `"auto_interv"`, or `"manual_interv"` |
| `--interv_factor`      | float    | 0.1             | Scaling factor for intervention                              |
| `--use_full_residual`  | flag     | False           | Whether to use full residual during modification             |
| `--token_num`          | int      | 20              | Number of top tokens to display during inference             |


 ## 🔍 Quick Semantic/Syntactic Analysis with ChatGPT

You can leverage **ChatGPT/DeepSeek** to quickly analyze MLP SVD directions and understand their semantic or syntactic patterns. Here’s how:

**Prompt Template:**

```
Please analyze all tokens provided in each direction to see if they have consistent semantics and functions. If so, summarize the possible semantics or functions for each direction using "Direction i + Consistency level (low/medium/high)  + Token type".  Write the results to a csv file.
```

**Input Example:**

```
Direction 1 POS:
,, -, the, and, ., in, a, to, (, first, at, time, ", new, of, two, on, all, or, so, as, :, that, G, current, actual, real, H, this, other, for, public, with, one, F

Direction 2 POS:
final, future, Holy, res, best, vast, beauty, great, most, od, trial, otted, safety, complete, new, det, grand, majority, rew, original, �, ア, specific, catch, ダ, cross, ro, heart, same, lowest, continuous, Great, weekly, least, Build

Direction 3 POS:
FDA, cement, recomb, decimal, goalt, azeera, TEAM, EQ, Geoff, bilateral, CHAT, VALUE, ', //, initials, Rare, CW, Geographic, catalog, crit, ée, partName, patented, repl, NCAA, interpersonal, ilateral, Paste, Sims, ˈ, Logged, Commercial, carbs, innov, isoft

Direction 4 POS:
cffffcc, respawn, CVE, voic, catentry, natureconservancy, reminis, dehuman, emot, obook, motto, Season, ndra, Shares, Niet, jerseys, recol, gunned, antagon, orem, Bot, Volunte, badges, kid, wo, pestic, REPL, inconven, Pokemon, Schedule, Pokémon, inciner, GMT, distrust, folios
```

**Usage:**

1. Copy the prompt and the top tokens for each direction into ChatGPT/Deepseek. You can input up to **100 directions** at once for analysis.
2. ChatGPT/DeepSeek will classify whether tokens in each direction share semantic meaning (e.g., entities, numbers, sentiment) or syntactic roles (e.g., articles, prepositions, verbs).
3. It can also provide an estimated proportion of tokens within each direction that align with a common function or category.

**Example Output (abridged):**

* **Direction 1:** high – function words / basic grammar
* **Direction 2:** medium – adjectives / evaluative terms
* **Direction 3:** medium – named entities / proper nouns / abbreviations
* **Direction 4:** medium – specialized nouns / entities / game/culture references
* **Direction 5:** medium – proper nouns / domain-specific terms
* **Direction 6:** medium – technical / gaming / organizational nouns
* **Direction 7:** high – positive adjectives / adverbs

