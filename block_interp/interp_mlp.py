import os
from tqdm.auto import tqdm
from block_interp.model_load import load_model_and_embeddings, get_mlp_matrices
from block_interp.top_tok import top_tokens
from block_interp.mlp_svd_utils import compute_svd, reshape_emb_matrix


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
        with_values: bool = True,
        only_english: bool = False,
        only_ascii: bool = True
    ):
        """
        Extract top-k positively correlated tokens for a given score vector.

        Args:
            scores (torch.Tensor): Token score vector.
            topk (int): Number of top tokens to extract.
            with_values (bool): Whether to include numeric values.
            only_english (bool): Restrict output tokens to English.
            only_ascii (bool): Restrict output tokens to ASCII.

        Returns:
            list[str] or list[(str, float)]: Top tokens.
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
            only_ascii=only_ascii,
            with_values=with_values
        )


    def _save_singular_vectors_to_file(
        self,
        sign: str,
        vectors,
        topk_tokens: int,
        output_file: str,
        with_values: bool = True,
        only_english: bool = False,
        only_ascii: bool = True
    ):
        """
        Save singular vectors and their top tokens to a text file.

        Args:
            sign (str): "positive" or "negative".
            vectors (list[torch.Tensor]): List of direction score vectors.
            topk_tokens (int): Number of top tokens per direction.
            output_file (str): Path to output text file.
            with_values (bool): Include token values.
        """
        
        print(f"Saving {sign} singular vectors to {output_file} ...")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")

            for idx, vec in tqdm(
                enumerate(vectors),
                total=len(vectors),
                desc=f"Processing {sign} vectors",
            ):
                top_tok = self._extract_top_tokens(
                    vec.cpu(),
                    topk=topk_tokens,
                    with_values=with_values,
                    only_english=only_english,
                    only_ascii=only_ascii
                )

                if len(top_tok) == 1 and isinstance(top_tok[0], str):
                    top_tok_str = top_tok
                else:
                    top_tok_str = (
                        [f"{t}({v:.6f})" for t, v in top_tok]
                        if with_values else top_tok
                    )

                f.write(f"Direction {idx + 1}:\n")
                f.write(", ".join(top_tok_str))
                f.write("\n\n")

 
    def mlp_subspace_interp(
        self,
        layer_idx: int,
        out_dir: str = "result",
        topk_tokens: int = 50,
        topk_subspaces: int = 100,
        weight_type: str = "c_proj",
        interp_type: str = "effector",
        with_negative: bool = False,
        use_activation: bool = False,
        with_values: bool = True,
        only_english: bool = False,
        only_ascii: bool = True
    ):
        """
        Compute and interpret the top singular directions of an MLP layer.

        Args:
            layer_idx (int): Target layer index.
            out_dir (str): Directory to save results.
            topk_tokens (int): Top tokens per direction.
            topk_subspaces (int): Top singular directions.
            weight_type (str): "c_proj" or "c_fc".
            interp_type (str): "effector" or "detector".
            with_negative (bool): Save also the negative directions.
            use_activation (bool): Whether to include activation projection.
            with_values (bool): Include scores in output.
        """
        
        # Load weights and perform SVD  
        c_fc, c_proj, act = get_mlp_matrices(self.model, layer_idx)
        U, S, V = compute_svd(weight_type, c_fc=c_fc, c_proj=c_proj)

        # output path 
        output_file = os.path.join(
            out_dir,
            f"{self.model_name}_MLP_{weight_type}_{interp_type}",
            f"layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens.txt"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            os.remove(output_file)

        # Compute projections  
        if (
            (weight_type == "c_proj" and interp_type == "effector")
            or (weight_type == "c_fc" and interp_type == "detector")
        ):
            # Project V into embedding space
            pos_score = [
                V[i, :].float() @ self.W_emb.T
                for i in range(min(topk_subspaces, S.shape[0]))
            ]

        elif (
            (weight_type == "c_proj" and interp_type == "detector")
            or (weight_type == "c_fc" and interp_type == "effector")
        ):
            # Use reshaped embedding via activation
            reshape_matrix = reshape_emb_matrix(
                self.W_emb,
                c_fc,
                act,
                use_activation
            )
            pos_score = [
                reshape_matrix @ (S[i] * U[:, i])
                for i in range(min(topk_subspaces, S.shape[0]))
            ]

        else:
            raise ValueError(
                f"Unknown combination: weight_type={weight_type}, interp_type={interp_type}"
            )

        self._save_singular_vectors_to_file(
            "positive",
            pos_score,
            topk_tokens,
            output_file,
            with_values,
            only_english,
            only_ascii
        )


        if with_negative:
            neg_score = [-v for v in pos_score]
            self._save_singular_vectors_to_file(
                "negative",
                neg_score,
                topk_tokens,
                output_file,
                with_values,
                only_english,
                only_ascii
            )

        print(
            f"\n>> Analysis complete for '{weight_type}' "
            f"results saved to:\n{output_file}\n"
        )
