import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ke.edit import edit_mlp_layers
from block_interp.model_load import load_model_and_embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-medium"
model, tokenizer, W_E = load_model_and_embeddings(model_name, device)


# Parameters for editing
layers_to_edit = [16, 17, 18, 19, 20, 21, 22]
input_text = "The cat looks very"
original_target = " happy"
new_target = " cute"
weight_type = "c_proj"



# Next-token predictions
def get_next_token_predictions(model, tokenizer, text, device, top_k=10):
    """Return top-k next-token predictions and their probabilities."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
        tokens = [tokenizer.decode([i.item()]) for i in top_idx[0]]
        scores = top_vals[0].tolist()
    return list(zip(tokens, scores))


# Full text generation
def generate_full_text(model, tokenizer, text, device, max_new_tokens=30):
    """Generate a sequence of text from a prompt deterministically (top_k=1)."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=1
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# Display next-token predictions
def display_predictions(preds, title="Next-token prediction"):
    print(f" [{title}]")
    print("-" * 45)
    for token, score in preds:
        print(f"{token:<10}  {score:.4f}")
    print("-" * 45)



# Compare before vs after predictions
def compare_predictions(before, after, top_k=10):
    print("Top-10 next-token probability difference:")
    print("-" * 45)
    before_dict = {t: s for t, s in before}
    after_dict = {t: s for t, s in after}
    all_tokens = set(before_dict.keys()) | set(after_dict.keys())

    for t in all_tokens:
        before_s = before_dict.get(t, 0.0)
        after_s = after_dict.get(t, 0.0)
        diff = after_s - before_s
        mark = "⬆️" if diff > 0 else ("⬇️" if diff < 0 else "→")
        print(f"{t:<10}  {before_s:>6.4f} → {after_s:>6.4f}   ({diff:+.4f}) {mark}")
    print("-" * 45)



# Baseline inference before editing
print(" Running baseline inference...\n")
baseline_topk = get_next_token_predictions(model, tokenizer, input_text, device)
display_predictions(baseline_topk, f"Before: \"{input_text}\"")

baseline_text = generate_full_text(model, tokenizer, input_text, device)
print("\n [Before] Full generation:")
print(baseline_text)
print("\n" + "="*60 + "\n")



# Run MLP editing
edited_weights = edit_mlp_layers(
    W=model,
    model_name=model_name,
    in_seq=input_text,
    ori_target=original_target,
    new_target=new_target,
    layers=layers_to_edit,
    weight_type=weight_type,
    delta_boost=10,
    delta_suppress=2,
    device="cuda",
    interp_type="all",
    circuit_mode="DeEf",
    topk_subspaces=15,
    output_dir="result/ke"
)

# Apply edited weights to model
for layer_idx, W_edited in edited_weights.items():
    mlp = model.transformer.h[layer_idx].mlp
    if weight_type == "c_proj":
        mlp.c_proj.weight.data.copy_(W_edited)
        print(f" Edited layer {layer_idx} (c_proj)")
    elif weight_type == "c_fc":
        mlp.c_fc.weight.data.copy_(W_edited)
        print(f" Edited layer {layer_idx} (c_fc)")
    else:
        print(f" Unsupported weight type: {weight_type}")

print("\n MLP layers edited successfully.\n")


# Inference after editing
print("Running inference after editing...\n")
edited_topk = get_next_token_predictions(model, tokenizer, input_text, device)
display_predictions(edited_topk, f"After: \"{input_text}\"")

edited_text = generate_full_text(model, tokenizer, input_text, device)
print("\n[After] Full generation:")
print(edited_text)
print("\n" + "="*60 + "\n")



# Compare before vs after
compare_predictions(baseline_topk, edited_topk)

print("\n Done! You can now visually compare before vs after.")
