# NaNA: SVD-Based MLP Interpretability for Transformers

NaNA is a mechanistic interpretability framework for transformer language models. It decomposes MLP weight matrices via SVD into orthogonal **subspaces**, interprets each subspace as a (detector, effector) pair, and traces which subspaces across layers causally drive a specific next-token prediction.

 

## 🔍 Core Idea

Every MLP layer contains a weight matrix that can be decomposed as:

```
W  =  U · diag(S) · Vᵀ
```

Each rank-1 component `(u_k, σ_k, v_k)` defines a **subspace**:

- **Detector** (`v_k`): how strongly does the current hidden state activate this direction?  
  Score = alignment of layer input with `v_k` (for `c_proj`) or with the LN-weighted `c_fc` direction.
- **Effector** (`u_k`): which vocabulary tokens does this direction push toward?  
  Score = cosine similarity of `u_k` with the token embedding matrix.
- **Directional contribution** = detector × effector — ranks subspaces by their causal relevance to a target token.

The framework has three modes of analysis:

| Mode | Script | Question answered |
|---|---|---|
| **Interpretation** | `run_interp.py` | What semantic concept does each subspace encode? |
| **Circuit discovery** | `run_circuit.py` | Which subspaces across all layers predict token T from input X? |
| **Intervention** | `run_modify.py` | What happens to the prediction if we ablate or amplify a subspace? |
 
## 📂 Repository Structure

```
├── scripts/
│   ├── run_interp.py        # Subspace interpretation
│   ├── run_circuit.py       # Circuit discovery
│   └── run_modify.py        # Causal intervention
├── block_interp/            # Core analysis library
│   ├── interp.py            # SVD decomposition and token alignment
│   ├── circuit.py           # Detector/effector scoring and circuit extraction
│   └── modify.py            # Forward-hook-based weight intervention
├── dataset/                 # Dataset utilities
├── sae_training/            
├── plot_utils/               
├── vocab/                    
├── manual_interv_info/      # Hand-specified subspace intervention configs            
└── requirements.txt
```

 

## 📦 Installation

```bash
pip install -r requirements.txt
```
  
## 🧩 Example Usage

### 🔹 Subspace Interpretation

Identify which tokens each singular direction is most aligned with.

For up-projection matrix
```bash
python scripts/run_interp.py  \
    --model_name "gpt2" \
    --layers 7  \
    --topk_tokens 20  \
    --topk_subspaces 12 \
    --weight_type c_fc \
    --interp_type detector \
    --with_negative \
    --save_file \
    --out_dir results \
    --return_heatmap
```
For down-projection matrix
```bash
python scripts/run_interp.py \
  --model_name "gpt2-medium" \
  --layers 16 \
  --topk_tokens 20 \
  --topk_subspaces 12 \
  --weight_type c_proj \
  --interp_type effector \
  --with_negative \
  --save_file \
  --out_dir results \
  --return_heatmap
```

### 🔹 Circuit Discovery

Given an input sequence and a target token, rank every subspace in every layer by its directional contribution to that prediction.

```bash
python scripts/run_circuit.py \
  --model_name "gpt2-medium" \
  --gpu 0 \
  --topk_subspaces 50 \
  --weight_type "c_proj" \
  --circuit_mode "DeEf" \
  --interp_type "effector" \
  --output_dir "results" \
  --layers "all" \
  --in_seq "The cat looks very" \
  --target_word " happy"
```
 
 
### 🔹 Causal Intervention

> **Note:** Run `run_circuit.py` first to generate `circuit_points_scores_{weight_type}_{model_name}.json` for analysis.

Pathway Standard Experiment: Rebuild using only top-K subspaces
```bash
python scripts/run_modify.py \
  --model "gpt2-medium" \
  --weight_type c_proj \
  --auto_subspace_file "results/gpt2-medium_circuit/MLP_c_proj/DeEf/circuit_points_scores_c_proj_gpt2-medium.json" \
  --input_text "The cat looks very" \
  --modify_type rebuild \
  --token_num 15 \
  --layers 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --gene_or_abla general \
  --top_subspaces 5 \
  --output_dir results \
  --use_full_residual
```

Pathway Ablation Experiment: Remove top-K subspaces
```bash
python scripts/run_modify.py \
  --model "gpt2-medium" \
  --weight_type c_proj \
  --auto_subspace_file "results/gpt2-medium_circuit/MLP_c_proj/DeEf/circuit_points_scores_c_proj_gpt2-medium.json" \
  --input_text "The cat looks very" \
  --modify_type rebuild \
  --token_num 15 \
  --layers 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --gene_or_abla ablation \
  --top_subspaces 5 \
  --output_dir results \
  --use_full_residual 
```

Single-layer Subspace Intervention: Manually specified subspace intervention
```bash
python scripts/run_modify.py \
  --model "gpt2-medium" \
  --weight_type c_proj \
  --manual_subspace_file "manual_interv_info/info_c_proj_gpt2-medium_L16SP7.json" \
  --input_text "The cat looks very" \
  --modify_type manual_interv \
  --token_num 15 \
  --layers 16 \
  --gene_or_abla general \
  --output_dir results \
  --use_full_residual \
  --interv_factor -7
```
### ⚙️ Configuration Parameters

The following are commonly used configuration options:
| Argument | Default | Description |
|---|---|---|
| `--model_name` | `gpt2-medium` | HuggingFace model name |
| `--layers` | `all` | Layer indices to analyse, or `all` |
| `--weight_type` | `c_proj` | `c_proj` (output projection) or `c_fc` (input projection + LN) |
| `--interp_type` | `all` | `detector`, `effector`, or `all` |
| `--topk_tokens` | `10` | Top tokens to report per subspace |
| `--topk_subspaces` | `50` | Number of subspaces to report per layer |
| `--with_negative` | — | Also report negatively aligned tokens |
| `--return_heatmap` | — | Save direction × token heatmaps as PNG |
| `--save_file` | — | Write JSON results to disk |
| `--in_seq` | required | Input text |
| `--target_word` | required | Token whose prediction circuit is traced |
| `--circuit_mode` | `DeEf` | `DeEf` (both), `De` (detector only), `Ef` (effector only) |
| `--topk_subspaces` | `15` | Top subspaces to include in the circuit |
| `--do_interp` | — | Also run token-level interpretation on selected subspaces |
| `--use_abs_contribute` | — | Rank by absolute contribution value |
| `--modify_type` | required | `rebuild`, `auto_interv`, or `manual_interv` |
| `--top_subspaces` | `10` | Subspaces to keep/scale (for `rebuild` and `auto_interv`) |
| `--interv_factor` | `1.0` | Scale factor applied to selected subspaces |
| `--gene_or_abla` | `general` | `general` (amplify) or `ablation` (suppress) |
| `--use_positive_only` | — | Restrict to positively contributing subspaces |
| `--token_num` | `20` | Number of generated tokens to compare |

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
 