 
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
├── MLP_svd_detector.py          # Analyze SVD input directions of MLP weight (detector)
├── MLP_svd_effector.py          # Analyze SVD output directions of MLP weight (effector)
├── run_svd_probe.py             # Perform subspace-level interventions within the model
├── circuit/                     # Circuit-level analysis modules
│   ├── __init__.py
│   ├── detector.py
│   └── effector.py
├── svd_probe/                   # Core SVD analysis and intervention utilities
│   ├── __init__.py
│   ├── svd_ops.py               # SVD computation and subspace projections
│   ├── model_interface.py       # Model loading, token generation, layer IO collection
│   ├── subspace_intervention.py # Functions to enhance or remove subspace directions
│   └── circuit_analysis.py      # Detector, effector, and DeEf circuit analysis
├── result/                      # Example outputs and top tokens per subspace
├── requirements.txt             # Python dependencies

```
 


## 📦 Requirements

All dependencies are listed in `requirements.txt`.
To install:

```bash
pip install -r requirements.txt
```


## 🧩 Example Usage

```bash
# Analyze detector (input) side
python MLP_svd_detector.py

# Analyze effector (output) side
python MLP_svd_effector.py
```

Results will be saved under:

```
result_svd_detector/
result_svd_effector/
```

Each file lists the top tokens associated with each SVD direction, making it easy to interpret semantic axes.



### Configuration Parameters

Both scripts share similar configurable options:

| Argument         | Description                         | Default         |
| ---------------- | ----------------------------------- | --------------- |
| `model_name`     | Hugging Face model name             | `"gpt2-medium"` |
| `layers_to_use`  | List of layers to analyze           | `[16]`          |
| `topk_tokens`    | Top-k tokens per direction          | `35`            |
| `topk_subspaces` | Number of singular directions       | `500`           |
| `with_negative`  | Whether to show negative directions | `False`         |



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

