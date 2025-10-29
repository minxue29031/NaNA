 
# 🧠 Transformers SVD Analysis — Detector & Effector

This repository provides tools for analyzing the semantic subspaces of MLP layers in transformer-based language models (e.g., "gpt2-medium"). By decomposing transformer blocks (e.g., MLPs) into sums of rank-1 subspaces: $W = \sum_{i=1}^{\text{rank}(W)} \sigma_i \  u_i \  v_i^T$, we can extract **detector** vectors ($u_i$) and **effector** vectors ($v_i^T$). These vectors reveal interpretable directions in the embedding space that correspond to meaningful linguistic or conceptual patterns.

## 🔍 Concept Overview

In Transformer MLP blocks,the **SVD decomposition** of these weight matrices reveals interpretable "directions" in hidden space:

* **Detector directions:** The input directions of the W_out subspace. Computes similarity with the embedding matrix to return the most highly correlated tokens.
* **Effector directions:** The output directions of the W_out subspace. Computes similarity with the embedding matrix to return the most highly correlated tokens.
  
By inspecting top tokens aligned with each singular direction, we can identify **semantic features** (e.g., sentiment, number, tense, named entities, etc.) captured by each MLP layer.



## 📂 Repository Structure

```
MLP_SVD_Project/
├── run_interp.py                # Analyze SVD directions of MLP weights
├── run_circuit.py               # Perform subspace-level circuit analysis
├── block_interp/                # Core modules for MLP SVD interpretation
│   ├─ top_tok.py                # Token processing and top-k extraction
│   ├─ interp_mlp.py             # Core class for MLP subspace analysis (MLP_DEEF_INTERP)
│   ├─ mlp_svd_utils.py          # SVD computation and embedding matrix reshaping utilities
│   ├─ model_load.py             # Model loading and MLP weight extraction
│   └─ __init__.py
├── circuit/                     # SVD analysis and intervention utilities
│   ├─ svd_ops.py                # SVD computation and subspace projections
│   ├─ collect_circuit_info.py   # Gather info for circuit analysis
│   ├─ model_interface.py        # Model loading, token generation, and layer IO collection
│   ├─ subspace_intervention.py  # Functions to enhance or remove subspace directions
│   ├─ circuit_analysis.py       # Detector, effector, and DeEf circuit analysis
│   └─ __init__.py
├── plot_utils/                  # Visualization utilities
│   ├─ plot_heatmap.py           # Heatmaps of subspace activated tokens
│   ├─ plot_path.py              # Subspace contribution flow visualization
│   └─ plot_subspace_contribute.py # Visualize subspace contributions per token
├── result/                      # Example outputs and top tokens per subspace
├── requirements.txt             # todu
```
 


## 📦 Requirements

All dependencies are listed in `requirements.txt`.
To install:

```bash
pip install -r requirements.txt
```

## 🧩 Example Usage

### 🔹 Detector & Effector Analysis

```bash
python run_interp.py \
    --model_name gpt2-medium \
    --layers 16 17 \
    --out_dir result \
    --topk_tokens 10 \
    --topk_subspaces 50 \
    --weight_type c_proj \
    --interp_type detector \
    --with_negative \
    --use_activation \
    --with_values
```

These scripts analyze **SVD directions** in the MLP layers of a transformer model.
Each produces a ranked list of **top tokens** most associated with each singular direction, making it easier to interpret **semantic axes** within the model.


### 🔹 Subspace Intervention

```bash
python run_svd_probe.py
```

Performs **causal interventions** on specific MLP subspace directions — either **enhancing** or **removing** them — to study how these directions
affect the model’s predictions and output semantics.

### 📊 Output

Each script saves its results under the `result/` directory, including:

* **Top tokens** associated with each SVD direction
* **Activation strengths** for selected subspaces
* **Modified generations** after subspace intervention
 
### ⚙️ Configuration Parameters

Both scripts share similar configurable options:

| Parameter          | Type     | Default         | Description                                                              |
| ------------------ | -------- | --------------- | ------------------------------------------------------------------------ |
| `--model_name`     | str      | `"gpt2-medium"` | Hugging Face model name. Options: `"gpt2"`, `"gpt2-medium"`, `"gpt2-xl"` |
| `--layers`         | int list | `[16]`          | Layer indices to analyze (space-separated)                               |
| `--out_dir`        | str      | `"result"`      | Directory to save results                                                |
| `--topk_tokens`    | int      | `10`            | Top-K tokens per direction                                               |
| `--topk_subspaces` | int      | `50`            | Number of top singular directions to analyze                             |
| `--weight_type`    | str      | `"c_proj"`      | MLP weight type: `c_proj`, `c_fc`, or `ov(TODO)`                               |
| `--interp_type`    | str      | `"detector"`    | Interpretation type: `detector` or `effector`                            |
| `--with_negative`  | bool     | `False`         | Save negative directions as well                                         |
| `--use_activation` | bool     | `False`         | Apply activation function in projection                                  |
| `--with_values`    | bool     | `False`         | Include token scores in output                                           |


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

