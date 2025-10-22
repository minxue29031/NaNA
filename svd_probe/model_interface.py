import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, device):
    print(f"======= Loading Model {model_name} ===========")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_next_token(model, tokenizer, text, device, max_new_tokens=1):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits   
        last_token_logits = logits[:, -1, :]   
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1)
        next_token = tokenizer.decode(next_token_id)
        
    print(f"\n >> Next token: {next_token} (ID: {next_token_id.item()})")
    
    return next_token, next_token_id.item()


def collect_layer_input_output(model, tokenizer, layer_idx, text, device):
    layer_inputs, layer_outputs = [], []

    def hook_fn(module, input, output):
        layer_inputs.append(input[0][:, -1, :].detach().cpu())
        layer_outputs.append((output[:, -1, :].detach().cpu()))

    block = model.transformer.h[layer_idx]
    hook = block.mlp.c_proj.register_forward_hook(hook_fn)

    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    hook.remove()

    layer_input = torch.cat(layer_inputs, dim=0)
    layer_output = torch.cat(layer_outputs, dim=0)

    print(f"Layer {layer_idx} input shape: {layer_input.shape}")
    print(f"Layer {layer_idx} output shape: {layer_output.shape}")

    return {"input": layer_input, "output": layer_output}, inputs["input_ids"]

 
def project_mlp_acv_to_vocab(model, tokenizer, layer_output, topk=20):
    W_E = model.get_input_embeddings().weight.detach().cpu() 
    logits_proj = layer_output @ W_E.T  
    top_vals, top_idx = torch.topk(logits_proj, topk, dim=-1) 

    top_tokens_list = []
    for batch_idx in range(layer_output.shape[0]):
        tokens = [tokenizer.decode([idx.item()]) for idx in top_idx[batch_idx]]
        scores = top_vals[batch_idx].tolist()
        top_tokens_list.append(list(zip(tokens, scores)))

    print(f"\n >> Ori top-{topk} tokens by projecting MLP output:")
    for token, score in top_tokens_list[0]:
        print(f"  {token}: {score:.6f}")

    return top_tokens_list

