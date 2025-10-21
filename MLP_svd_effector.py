import torch
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from copy import deepcopy
from tqdm.auto import tqdm, trange
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from copy import deepcopy
from torch.nn import functional as F
from tabulate import tabulate
from tqdm import tqdm, trange
import functools
import math
import site
site.main()
 

def keep_k(x, k=100, absolute=True, dim=-1):
    shape = x.shape
    x_ = x
    if absolute:
        x_ = abs(x)
    values, indices = torch.topk(x_, k=k, dim=dim)
    res = torch.zeros_like(x)
    res.scatter_(dim, indices, x.gather(dim, indices))
    return res

def get_max_token_length(tokens):
  maxlen = 0
  for t in tokens:
    l = len(t)
    if l > maxlen:
      maxlen = l
  return maxlen

def pad_with_space(t, maxlen):
  spaces_to_add = maxlen - len(t)
  for i in range(spaces_to_add):
    t += " "
  return t

def convert_to_tokens(indices, tokenizer, extended, extra_values_pos, strip=True, pad_to_maxlen=False):
    if extended:
        res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else 
               (f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx-extra_values_pos}]") 
               for idx in indices]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    if pad_to_maxlen:
      maxlen = get_max_token_length(res)
      res = list(map(lambda t: pad_with_space(t, maxlen), res))
    return res


def top_tokens(v_tok, k=100, tokenizer=None, only_english=False, only_ascii=True, with_values=False, 
               exclude_brackets=False, extended=True, extra_values=None, pad_to_maxlen=False):
    if tokenizer is None:
        tokenizer = my_tokenizer
    v_tok = deepcopy(v_tok)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
    if only_english: 
        ignored_indices =[key for val, key in tokenizer.vocab.items() if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    v_tok[ignored_indices] = -np.inf
    extra_values_pos = len(v_tok)
    if extra_values is not None:
        v_tok = torch.cat([v_tok, extra_values])
    values, indices = torch.topk(v_tok, k=k)
    res = convert_to_tokens(indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos,pad_to_maxlen = pad_to_maxlen)
    if with_values:
        res = list(zip(res, values.cpu().numpy()))
    return res
 
 
def get_model_tokenizer_embedding(model_name):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cpu':
    print("WARNING: you should probably restart on a GPU runtime")

  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  emb = model.get_output_embeddings().weight.data.T.detach()
  return model, tokenizer, emb, device


def get_model_info(model):
  num_layers = model.config.n_layer
  num_heads = model.config.n_head
  hidden_dim = model.config.n_embd
  head_size = hidden_dim // num_heads
  return num_layers, num_heads, hidden_dim, head_size

def get_mlp_weights(model,num_layers, hidden_dim):
  Ks = []
  Vs = []
  for j in range(num_layers):
    K = model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T.detach()
    ln_2_weight = model.get_parameter(f"transformer.h.{j}.ln_2.weight").detach()
    K = torch.einsum("oi,i -> oi", K, ln_2_weight)
    
    V = model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
    Ks.append(K)
    Vs.append(V)
  
  Ks =  torch.cat(Ks)
  Vs = torch.cat(Vs)
  K_heads = Ks.reshape(num_layers, -1, hidden_dim)
  V_heads = Vs.reshape(num_layers, -1, hidden_dim)
  return K_heads, V_heads

def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
  qkvs = []
  for j in range(num_layers):
    qkv = model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight").detach().T
    ln_weight_1 = model.get_parameter(f"transformer.h.{j}.ln_1.weight").detach()
    
    qkv = qkv - torch.mean(qkv, dim=0) 
    qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
    qkvs.append(qkv.T)

  W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
  W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") for j in range(num_layers)]).detach()
  W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
  W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  return W_Q_heads, W_K_heads, W_V_heads, W_O_heads
 
 
def MLP_V_top_singular_vectors_Wout(layer_idx, all_tokens, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
    Vs_all = []

    with torch.no_grad():
        W_matrix = model.get_parameter(f"transformer.h.{layer_idx}.mlp.c_proj.weight").detach()
        U, S, Vval = torch.svd(W_matrix)

        Vs = []
        for i in range(topk_subspaces):
            acts = Vval.T[i, :].float() @ emb
            Vs.append(acts)
        Vs_all.append(("positive", Vs))

        if with_negative:
            Vs_neg = []
            for i in range(topk_subspaces):
                acts = -Vval.T[i, :].float() @ emb
                Vs_neg.append(acts)
            Vs_all.append(("negative", Vs_neg))

    with open(output_file, "w", encoding="utf-8") as f:
        for sign, Vs in Vs_all:
            f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
            for idx, vec in enumerate(Vs):
                top_tok = top_tokens(vec.cpu(), k=topk_tokens, pad_to_maxlen=True)
                f.write(f"Direction {idx+1}:\n")
                f.write(", ".join(top_tok))
                f.write("\n\n")

    print(f"Results saved to {output_file}")


def MLP_K_top_singular_vectors_Win(K, emb, layer_idx, all_tokens, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
    Vs_all = []

    W_matrix = K[layer_idx, :, :]
    U, S, V = torch.linalg.svd(W_matrix, full_matrices=False)
    
    print("------------emb.shape------------", emb.shape)
    print("------------V.shape------------", V.shape)
    Vs = [V[i, :].float() @ emb for i in range(topk_subspaces)]
    Vs_all.append(("positive", Vs))

    if with_negative:
        Vs_neg = [-V[i, :].float() @ emb for i in range(topk_subspaces)]
        Vs_all.append(("negative", Vs_neg))


    with open(output_file, "w", encoding="utf-8") as f:
        for sign, Vs in Vs_all:
            f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
            for idx, vec in enumerate(Vs):
                top_tok = top_tokens(vec.cpu(), k=topk_tokens)
                f.write(f"Direction {idx}:\n")
                f.write(", ".join(top_tok))
                f.write("\n\n")

    print(f"Results saved to {output_file}")


def OV_top_singular_vectors(W_V_heads, W_O_heads, emb, layer_idx, head_idx, all_tokens, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
    Vs_all = []
 
    W_V_tmp, W_O_tmp = W_V_heads[layer_idx, head_idx, :], W_O_heads[layer_idx, head_idx]
    OV = W_V_tmp @ W_O_tmp
    U, S, V = torch.linalg.svd(OV)
    Vs_pos = []
    for i in range(topk_subspaces):
        acts = V[i, :].float() @ emb
        Vs_pos.append(acts)
    Vs_all.append(("positive", Vs_pos))

    if with_negative:
        Vs_neg = []
        for i in range(topk_subspaces):
            acts = -V[i, :].float() @ emb
            Vs_neg.append(acts)
        Vs_all.append(("negative", Vs_neg))

 
    with open(output_file, "w", encoding="utf-8") as f:
        for sign, Vs in Vs_all:
            f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
            for idx, vec in enumerate(Vs):
                top_tok = top_tokens(vec.cpu(), k=topk_tokens, pad_to_maxlen=True)
                f.write(f"Direction {idx}:\n")
                f.write(", ".join(top_tok))
                f.write("\n\n")

    print(f"OV results saved to {output_file}")


if __name__ == "__main__":
    model_name="gpt2-medium"
    layers_to_use = [16]
    out_dir = "result"
    topk_tokens = 50
    topk_subspaces = 100
    model, tokenizer, emb, device = get_model_tokenizer_embedding(model_name)
    my_tokenizer = tokenizer
    num_layers, num_heads, hidden_dim, head_size = get_model_info(model)
    all_tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    
    K,V = get_mlp_weights(model, num_layers = num_layers, hidden_dim = hidden_dim)
    W_Q_heads, W_K_heads, W_V_heads, W_O_heads = get_attention_heads(model, num_layers=num_layers, hidden_dim=hidden_dim, num_heads=num_heads, head_size = head_size)

    run_mlp_Wout = True
    run_mlp_Win  = False
    run_OV = False
    print(f"Run flags -> MLP Wout: {run_mlp_Wout}, MLP Win: {run_mlp_Win}, OV: {run_OV}")

    
    for layer_idx in layers_to_use: 
        if run_mlp_Wout:
            os.makedirs(out_dir, exist_ok=True)
            MLP_V_top_singular_vectors_Wout(
                layer_idx=layer_idx,
                all_tokens=all_tokens,
                topk_tokens=topk_tokens,                
                topk_subspaces=topk_subspaces, 
                with_negative=True,    
                output_file=f"{out_dir}/MLPout_layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
            )
 
        if run_mlp_Win:
            os.makedirs(out_dir, exist_ok=True)
            MLP_K_top_singular_vectors_Win(
                K, 
                emb, 
                layer_idx, 
                all_tokens, 
                topk_tokens=topk_tokens, 
                topk_subspaces=topk_subspaces,
                with_negative=True, 
                output_file=f"{out_dir}/MLPin_layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
            )

        if run_OV: 
            os.makedirs(out_dir, exist_ok=True)
            for head_idx in range(num_heads):
                OV_top_singular_vectors(
                    W_V_heads, W_O_heads, emb, 
                    layer_idx=layer_idx, 
                    head_idx=head_idx,
                    all_tokens=all_tokens, 
                    topk_tokens=topk_tokens, 
                    topk_subspaces=topk_subspaces,
                    with_negative=True,
                    output_file=f"{out_dir}/OV_layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
                )
