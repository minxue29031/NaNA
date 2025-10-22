import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLPDirectionDetector:
    """
    Analyze MLP output subspaces in transformer layers using SVD,
    and identify top-correlated tokens for each direction.
    """

    def __init__(self, model_name: str, output_dir: str = "result", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.output_dir = output_dir

        self._load_model_and_tokenizer()


    def _load_model_and_tokenizer(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.W_emb = self.model.get_input_embeddings().weight.detach().to(self.device)
        print(f"Embedding matrix loaded: {self.W_emb.shape}")


    def _get_mlp_matrices(self, layer_idx):
        """Extract MLP matrices for a given transformer layer."""
        block = self.model.transformer.h[layer_idx]
        return block.mlp.c_fc, block.mlp.c_proj, block.mlp.act

    def _compute_scores_matrix(self, c_fc, act, use_activation=False):
        """Project embeddings through MLP input (c_fc)."""
        scores_matrix = self.W_emb @ c_fc.weight + c_fc.bias
        if use_activation:
            scores_matrix = act(scores_matrix)
        return scores_matrix

    def _perform_svd(self, matrix):
        """Perform SVD decomposition on projection weights."""
        return torch.linalg.svd(matrix.detach().cpu(), full_matrices=False)

    def _extract_top_tokens(self, scores, topk=20, positive=True):
        """Extract top-k tokens correlated (positive or negative) with a direction."""
        mask = scores > 0 if positive else scores < 0
        if not mask.any():
            return ["=== No valid tokens ==="], torch.tensor([float("nan")])

        masked_indices = torch.arange(scores.shape[0], device=self.device)[mask]
        vals, topk_idx = torch.topk(torch.abs(scores[mask]), min(topk, mask.sum()))
        idx = masked_indices[topk_idx]
        if not positive:
            vals = -vals
        tokens = [self.tokenizer.decode([int(i)]) for i in idx]
        return tokens, vals.cpu()

    def _analyze_single_layer(
        self, layer_idx, k=1024, topk_tokens=20, with_negative=False, use_activation=False
    ):
        """Compute SVD + token correlations for one layer."""
        c_fc, c_proj, act = self._get_mlp_matrices(layer_idx)
        scores_matrix = self._compute_scores_matrix(c_fc, act, use_activation)
        U, S, _ = self._perform_svd(c_proj.weight)

        all_results = []
        for i in tqdm(range(min(k, S.shape[0])), desc=f"Layer {layer_idx} SVD"):
            u_i = S[i] * U[:, i]
            scores = scores_matrix @ u_i.to(self.device)

            pos_tokens, pos_vals = self._extract_top_tokens(scores, topk_tokens, positive=True)
            neg_tokens, neg_vals = ([], [])
            if with_negative:
                neg_tokens, neg_vals = self._extract_top_tokens(scores, topk_tokens, positive=False)

            all_results.append({
                "direction": i + 1,
                "pos": list(zip(pos_tokens, pos_vals.tolist())),
                "neg": list(zip(neg_tokens, neg_vals.tolist())) if with_negative else []
            })

        return all_results


    def _save_results(self, layer_idx, results, k, topk, with_negative, print_scores):
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(
            self.output_dir, f"MLPout_layer{layer_idx}_top{k}subspaces_top{topk}tokens_detector.txt"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            for res in results:
                pos_line = ", ".join(
                    f"{t}({s:.6f})" if print_scores else t for t, s in res["pos"]
                )
                f.write(f"Direction {res['direction']} POS:\n{pos_line}\n\n")
                if with_negative and res["neg"]:
                    neg_line = ", ".join(
                        f"{t}({s:.6f})" if print_scores else t for t, s in res["neg"]
                    )
                    f.write(f"Direction {res['direction']} NEG:\n{neg_line}\n\n")
        print(f">> Results saved: {file_path}")


    def run_layer_analysis(
        self, layer_idx, k=100, topk_tokens=50,
        with_negative=False, use_activation=False, print_scores=False
    ):

        print(f"\n=== Running analysis on layer {layer_idx} ===")
        results = self._analyze_single_layer(
            layer_idx, k=k, topk_tokens=topk_tokens,
            with_negative=with_negative, use_activation=use_activation
        )
        self._save_results(layer_idx, results, k, topk_tokens, with_negative, print_scores)
