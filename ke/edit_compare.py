import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ke.edit import edit_mlp_layers
from block_interp.model_load import load_model_and_embeddings


class MLPEditor:
    def __init__(self, model_name="gpt2-medium", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.model, self.tokenizer, self.W_E = load_model_and_embeddings(
            model_name, self.device
        )

        self.baseline_topk = None
        self.baseline_text = None
        self.edited_topk = None
        self.edited_text = None

    # Utility methods
    def get_next_token_predictions(self, text, top_k=10):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
            tokens = [self.tokenizer.decode([i.item()]) for i in top_idx[0]]

        return list(zip(tokens, top_vals[0].tolist()))

    def generate_full_text(self, text, max_new_tokens=30):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=1,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def display_predictions(self, preds, title="Next-token prediction"):
        print(f"\n[{title}]")
        print("-" * 45)
        for token, score in preds:
            print(f"{token:<10}  {score:.4f}")
        print("-" * 45)

    def compare_predictions(self, before, after):
        print("\nTop-10 next-token probability differences (sorted by absolute change):")
        print("-" * 60)

        before_dict = {t: s for t, s in before}
        after_dict = {t: s for t, s in after}
        all_tokens = set(before_dict.keys()) | set(after_dict.keys())

        diffs = []
        for t in all_tokens:
            before_s = before_dict.get(t, 0.0)
            after_s = after_dict.get(t, 0.0)
            diff = after_s - before_s
            diffs.append((t, before_s, after_s, diff, abs(diff)))

        # Sort by absolute value
        diffs.sort(key=lambda x: x[4], reverse=True)

        for t, before_s, after_s, diff, _ in diffs:
            mark = "\u2191" if diff > 0 else ("\u2193" if diff < 0 else "\u2192")
            print(f"{t:<10}  {before_s:>6.4f} → {after_s:>6.4f}   ({diff:+.4f}) {mark}")

        print("-" * 60)


    def run_inference(self, text, title="Inference"):
        topk_preds = self.get_next_token_predictions(text)
        self.display_predictions(topk_preds, f"{title}: \"{text}\"")

        full_text = self.generate_full_text(text)
        print(f"\n[{title}] Full generation:\n{full_text}\n")

        return topk_preds, full_text

    # MLP Editing
    def edit_mlp(
        self,
        input_text,
        original_target,
        new_target,
        layers,
        weight_type="c_proj",
        delta_boost=0.8,
        delta_suppress=0.8,
        interp_type="all",
        circuit_mode="DeEf",
        topk_subspaces=15,
        output_dir="result/ke",
    ):
        edited_weights = edit_mlp_layers(
            W=self.model,
            model_name=self.model_name,
            in_seq=input_text,
            ori_target=original_target,
            new_target=new_target,
            layers=layers,
            weight_type=weight_type,
            delta_boost=delta_boost,
            delta_suppress=delta_suppress,
            device=self.device,
            interp_type=interp_type,
            circuit_mode=circuit_mode,
            topk_subspaces=topk_subspaces,
            output_dir=output_dir,
        )

        for layer_idx, W_edited in edited_weights.items():
            mlp = self.model.transformer.h[layer_idx].mlp
            if weight_type == "c_proj":
                mlp.c_proj.weight.data.copy_(W_edited)
            elif weight_type == "c_fc":
                mlp.c_fc.weight.data.copy_(W_edited)
            print(f"Edited layer {layer_idx} ({weight_type})")

        print("\nMLP layers edited successfully.\n")

    # Full pipeline
    def run_full_pipeline(
        self,
        input_text,
        layers_to_edit,
        original_target,
        new_target,
        weight_type,
        delta_boost,
        delta_suppress,
        interp_type,
        circuit_mode,
        topk_subspaces,
        output_dir,
    ):
        print("Running baseline inference...")
        self.baseline_topk, self.baseline_text = self.run_inference(
            input_text, "Before"
        )
        print("=" * 60)

        self.edit_mlp(
            input_text=input_text,
            original_target=original_target,
            new_target=new_target,
            layers=layers_to_edit,
            weight_type=weight_type,
            delta_boost=delta_boost,
            delta_suppress=delta_suppress,
            interp_type=interp_type,
            circuit_mode=circuit_mode,
            topk_subspaces=topk_subspaces,
            output_dir=output_dir,
        )

        print("Running inference after editing...")
        self.edited_topk, self.edited_text = self.run_inference(
            input_text, "After"
        )
        print("=" * 60)

        self.compare_predictions(self.baseline_topk, self.edited_topk)
        print("\nDone! You can now visually compare before vs after.")

 