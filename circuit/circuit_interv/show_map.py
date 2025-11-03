import os
import torch
import json


def run_inference(model, tokenizer, text, device, topk=10):
    """
    Run inference and print top-k predictions
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_last = outputs.logits[:, -1, :]
        top_vals, top_idx = torch.topk(logits_last, topk, dim=-1)

    tokens = [tokenizer.decode([i.item()]) for i in top_idx[0]]
    scores = top_vals[0].tolist()

    print(f"\nInput: {text}")
    print(f"Top-{topk} predictions:")
    for i, (tok, score) in enumerate(zip(tokens, scores), 1):
        print(f"{i:2d}. {tok!r} (logit={score:.4f})")

    token_score_list = [{"token": tok, "score": float(score)} for tok, score in zip(tokens, scores)]
    result = {"input_text": text, "predictions": token_score_list}

    return result




def show_infer(model, tokenizer, input_text, hooks, device, topk=10, save_dir=None):
    results = {}

    results["modified"] = run_inference(model, tokenizer, input_text, device, topk)
    print("\n<< The subspace modification operation is complete. >>")

    for h in hooks:
        h.remove()

    results["original"] = run_inference(model, tokenizer, input_text, device, topk)
    print("\n<< Original model operation is complete. >>")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "final_predictions.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


    
 
def print_top_tokens(tokenizer, y, weight_type, W_E=None, reshape_W_E=None, topk=10):
    with torch.no_grad():
        if weight_type == "c_fc":
            if reshape_W_E is None:
                raise ValueError("reshape_W_E must be provided for weight_type='c_fc'")
            logits = y @ reshape_W_E.T
        elif weight_type == "c_proj":
            if W_E is None:
                raise ValueError("W_E must be provided for weight_type='c_proj'")
            logits = y @ W_E.T
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        if logits.ndim == 3:
            logits = logits[:, -1, :]   

        top_vals, top_idx = torch.topk(logits, topk, dim=-1)
        top_idx_list = top_idx[0].tolist()
        top_vals_list = top_vals[0].tolist()

        print(f"\nTop-{topk} predictions:")
        for rank, (idx, score) in enumerate(zip(top_idx_list, top_vals_list), 1):
            token_str = tokenizer.decode([int(idx)])
            print(f"{rank:2d}. '{token_str}' (logit={score:.4f})")

        return [(tokenizer.decode([int(idx)]), float(score)) for idx, score in zip(top_idx_list, top_vals_list)]
