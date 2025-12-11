import os
import json
from tqdm.auto import tqdm
from block_interp.model_load import load_model_and_embeddings, get_mlp_matrices
from block_interp.top_tok import top_tokens
from block_interp.mlp_svd_utils import compute_svd, reshape_emb_matrix, parse_topk_subspaces
from plot_utils.plot_heatmap import plot_subspace_heatmap

class MLP_DEEF_INTERP:
    """
    Subspace-level interface for MLP SVD-based interpretation.
    Performs layer-wise decomposition of MLP weights, projects directions
    into embedding space, and identifies top-correlated tokens.
    """

    def __init__(self, model_name: str, output_dir: str = "result", device=None):
        self.device = device
        self.model_name = model_name
        self.output_dir = output_dir

        # Load model and embeddings
        self.model, self.tokenizer, self.W_emb = load_model_and_embeddings(
            model_name,
            device=device
        )
         
    def _extract_top_tokens(
        self,
        scores,
        topk: int,
        only_english: bool = False,
        only_ascii: bool = True
    ):
        """
        Extract top-k positively correlated tokens for a given score vector.
        """
        mask = scores > 0
        if not mask.any():
            return ["=== No valid tokens ==="]

        filtered_scores = scores.clone()
        filtered_scores[~mask] = -float("inf")

        return top_tokens(
            self.tokenizer,
            v_tok=filtered_scores,
            k=topk,
            only_english=only_english,
            only_ascii=only_ascii
        )
 
     
    def align_top_token_subspace(
        self,
        sign: str,
        vectors,
        layer_idx: int,
        topk_tokens: int,
        weight_type: str,
        interp_type: str,
        out_dir: str = None,
        only_english: bool = False,
        only_ascii: bool = True,
        real_indices: list = None,
        subspace_label: str = "all",
        save_file: bool = False,
    ):
    
        results = []
        for idx, vec in enumerate(tqdm(vectors, desc=f"Processing {sign} vectors")):
            top_tok = self._extract_top_tokens(
                vec.cpu(),
                topk=topk_tokens,
                only_english=only_english,
                only_ascii=only_ascii
            )

            token_entries = []
            for t in top_tok:
                if isinstance(t, tuple) and len(t) == 2:
                    token_entries.append({"token": t[0], "value": float(t[1])})
                else:
                    # fallback for single token or missing value
                    token_entries.append({"token": str(t), "value": None})

            direction_num = real_indices[idx] + 1 if real_indices else idx + 1
            results.append({
                "direction": direction_num,
                "top_tokens": token_entries
            })

        if sign.lower() == "negative":
            for entry in results:
                for t in entry["top_tokens"]:
                    if t["value"] is not None:
                        t["value"] = -t["value"]
                    
        if save_file:
            base_dir = out_dir or self.output_dir
            output_dir = os.path.join(
                base_dir,
                f"{self.model_name}_interp_dir", "data", f"MLP_{weight_type}_layer{layer_idx}"
            )
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(
                output_dir,
                f"{interp_type}_subspaces{subspace_label}_top{topk_tokens}tokens_{sign}.json"
            )

            print(f"Saving {sign} singular vectors to {output_file} ...")
            
            son_results = results
            
            json_data = {
                "model_name": self.model_name,
                "layer_idx": layer_idx,
                "sign": sign,
                "weight_type": weight_type,
                "interp_type": interp_type,
                "subspace_label": subspace_label,
                "topk_tokens": topk_tokens,
                "Interpretability": results
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

        return results

            
    def mlp_subspace_interp(
        self,
        layer_idx: int,
        out_dir: str = "result",
        topk_tokens: int = 50,
        topk_subspaces="top100",   
        weight_type: str = "c_proj",
        interp_type: str = "effector",  
        with_negative: bool = False,
        only_english: bool = False,
        only_ascii: bool = True,
        save_file: bool = False,
        return_heatmap: bool = True
    ):
        """
        Compute and interpret the top singular directions of an MLP layer.

        Args:
            topk_subspaces: can be
                - int (e.g. 50): top 50 subspaces
                - list[int] (e.g. [3,7,12]): specific indices
                - str "all" or "topN": use all subspaces or top N subspaces
            interp_type: "effector", "detector", or "all"
        """

        # Load weights and perform SVD
        c_fc, c_proj, ln_2, act = get_mlp_matrices(self.model, layer_idx)
        U, S, V = compute_svd(weight_type, c_fc=c_fc, c_proj=c_proj, ln_2=ln_2)
        all_results = {}

        # Determine which subspaces to analyze
        total_subspaces = S.shape[0]
        indices, subspace_label = parse_topk_subspaces(topk_subspaces, total_subspaces)
 

        interp_types = (
            ["effector", "detector"] if interp_type == "all" else [interp_type]
        )

        for current_interp_type in interp_types:
            print(f"\n>>> Running interpretation: {current_interp_type}")

            pos_score = []
            for i in indices:
                if current_interp_type == "effector":
                    if weight_type == "c_proj":
                        score = V[i, :].float() @ self.W_emb.T
                    elif weight_type == "c_fc":
                        reshape_matrix = reshape_emb_matrix(
                            self.W_emb, c_fc, ln_2, act, use_activation=False
                        )
                        score = U[:, i] @ reshape_matrix.T
                    else:
                        raise ValueError(f"Unknown weight_type: {weight_type}")
                elif current_interp_type == "detector":
                    if weight_type == "c_fc":
                        score = V[i, :].float() @ self.W_emb.T
                    elif weight_type == "c_proj":
                        reshape_matrix = reshape_emb_matrix(
                            self.W_emb, c_fc, ln_2, act, use_activation=True
                        )
                        score = U[:, i] @ reshape_matrix.T
                    else:
                        raise ValueError(f"Unknown weight_type: {weight_type}")
                else:
                    raise ValueError(f"Unknown interp_type: {current_interp_type}")

                pos_score.append(score)

            # positive direction
            pos_results = self.align_top_token_subspace(
                "positive",
                pos_score,
                layer_idx,
                topk_tokens,
                weight_type,
                current_interp_type,
                out_dir=out_dir,
                only_english=only_english,
                only_ascii=only_ascii,
                real_indices=indices,
                subspace_label=subspace_label,
                save_file=save_file,
            )
            
            if return_heatmap:
                plot_subspace_heatmap(
                    pos_results,
                    self.model_name,
                    layer_idx=layer_idx,
                    weight_type=weight_type,
                    interp_type=current_interp_type,
                    sign="positive",
                    out_dir=out_dir
                )


            # negative direction
            if with_negative:
                neg_results = self.align_top_token_subspace(
                    "negative",
                    [-v for v in pos_score],
                    layer_idx,
                    topk_tokens,
                    weight_type,
                    current_interp_type,
                    out_dir=out_dir,
                    only_english=only_english,
                    only_ascii=only_ascii,
                    real_indices=indices,
                    subspace_label=subspace_label,
                    save_file=save_file,
                )
                
                if return_heatmap:
                    plot_subspace_heatmap(
                        neg_results,
                        self.model_name,
                        layer_idx=layer_idx,
                        weight_type=weight_type,
                        interp_type=current_interp_type,
                        sign="negative",
                        out_dir=out_dir
                    )



            all_results[current_interp_type] = {
                "positive": pos_results,
                "negative": neg_results
            }

 

        return all_results
 
