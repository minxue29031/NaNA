import torch
import os
from copy import deepcopy
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class MLPDirectionEffector:
    def __init__(self, model_name, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            print("WARNING: using CPU, consider using GPU for speed")

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emb = self.model.get_output_embeddings().weight.data.T.detach()
        self.num_layers = self.model.config.n_layer
        self.num_heads = self.model.config.n_head
        self.hidden_dim = self.model.config.n_embd
        self.head_size = self.hidden_dim // self.num_heads
        self.my_tokenizer = self.tokenizer

    @staticmethod
    def keep_k(x, k=100, absolute=True, dim=-1):
        x_ = abs(x) if absolute else x
        values, indices = torch.topk(x_, k=k, dim=dim)
        res = torch.zeros_like(x)
        res.scatter_(dim, indices, x.gather(dim, indices))
        return res

    @staticmethod
    def get_max_token_length(tokens):
        return max(len(t) for t in tokens)

    @staticmethod
    def pad_with_space(t, maxlen):
        return t + " " * (maxlen - len(t))

    def convert_to_tokens(self, indices, extended=True, extra_values_pos=None, strip=True, pad_to_maxlen=False):
        if extended:
            res = [self.tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(self.tokenizer) 
                   else (f"[pos{idx-len(self.tokenizer)}]" if idx < extra_values_pos 
                         else f"[val{idx-extra_values_pos}]") 
                   for idx in indices]
        else:
            res = self.tokenizer.convert_ids_to_tokens(indices)
        if strip:
            res = [x[1:] if x[0] == 'Ġ' else "#" + x for x in res]
            
        if pad_to_maxlen:
            fill_tokens = [t for t in res if t.replace('#','').isalnum()]
            maxlen = self.get_max_token_length(fill_tokens) if fill_tokens else 0
            res = [self.pad_with_space(t, maxlen) if t.replace('#','').isalnum() else t for t in res]

        return res

    def top_tokens(self, v_tok, k=100, only_english=False, only_ascii=True, with_values=False, 
                   exclude_brackets=False, extended=True, extra_values=None, pad_to_maxlen=False):
        v_tok = deepcopy(v_tok)
        ignored_indices = []
        if only_ascii:
            ignored_indices = [key for val, key in self.tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
        if only_english:
            ignored_indices += [key for val, key in self.tokenizer.vocab.items() if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
        if exclude_brackets:
            ignored_indices = set(ignored_indices).intersection(
                {key for val, key in self.tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
            ignored_indices = list(ignored_indices)
        v_tok[ignored_indices] = -float('inf')
        extra_values_pos = len(v_tok)
        if extra_values is not None:
            v_tok = torch.cat([v_tok, extra_values])
        values, indices = torch.topk(v_tok, k=k)
        res = self.convert_to_tokens(indices, extended=extended, extra_values_pos=extra_values_pos, pad_to_maxlen=pad_to_maxlen)
        if with_values:
            res = list(zip(res, values.cpu().numpy()))
        return res

     

    def MLP_V_top_singular_vectors_Wout(self, layer_idx, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
        Vs_all = []

        with torch.no_grad():
            W_matrix = self.model.get_parameter(f"transformer.h.{layer_idx}.mlp.c_proj.weight").detach()
            U, S, Vval = torch.svd(W_matrix)

            Vs = []
            for i in range(topk_subspaces):
                acts = Vval.T[i, :].float() @ self.emb
                Vs.append(acts)
            Vs_all.append(("positive", Vs))

            if with_negative:
                Vs_neg = []
                for i in range(topk_subspaces):
                    acts = -Vval.T[i, :].float() @ self.emb
                    Vs_neg.append(acts)
                Vs_all.append(("negative", Vs_neg))

        with open(output_file, "w", encoding="utf-8") as f:
            for sign, Vs in Vs_all:
                f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
                for idx, vec in tqdm(enumerate(Vs), total=len(Vs), desc=f"Processing {sign} vectors"):
                    top_tok = self.top_tokens(vec.cpu(), k=topk_tokens, pad_to_maxlen=True)
                    f.write(f"Direction {idx+1}:\n")
                    f.write(", ".join(top_tok))
                    f.write("\n\n")

        print(f"Results saved to {output_file}")


    def MLP_K_top_singular_vectors_Win(self, K, layer_idx, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
        Vs_all = []

        W_matrix = K[layer_idx, :, :]
        U, S, V = torch.linalg.svd(W_matrix, full_matrices=False)

        Vs = [V[i, :].float() @ self.emb for i in range(topk_subspaces)]
        Vs_all.append(("positive", Vs))

        if with_negative:
            Vs_neg = [-V[i, :].float() @ self.emb for i in range(topk_subspaces)]
            Vs_all.append(("negative", Vs_neg))


        with open(output_file, "w", encoding="utf-8") as f:
            for sign, Vs in Vs_all:
                f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
                for idx, vec in tqdm(enumerate(Vs), total=len(Vs), desc=f"Processing {sign} vectors"):
                    top_tok = self.top_tokens(vec.cpu(), k=topk_tokens)
                    f.write(f"Direction {idx}:\n")
                    f.write(", ".join(top_tok))
                    f.write("\n\n")

        print(f"Results saved to {output_file}")


    def OV_top_singular_vectors(self, W_V_heads, W_O_heads, layer_idx, head_idx, topk_tokens=20, topk_subspaces=10, with_negative=False, output_file=None):
        Vs_all = []
     
        W_V_tmp, W_O_tmp = W_V_heads[layer_idx, head_idx, :], W_O_heads[layer_idx, head_idx]
        OV = W_V_tmp @ W_O_tmp
        U, S, V = torch.linalg.svd(OV)
        Vs_pos = []
        for i in range(topk_subspaces):
            acts = V[i, :].float() @ self.emb
            Vs_pos.append(acts)
        Vs_all.append(("positive", Vs_pos))

        if with_negative:
            Vs_neg = []
            for i in range(topk_subspaces):
                acts = -V[i, :].float() @ self.emb
                Vs_neg.append(acts)
            Vs_all.append(("negative", Vs_neg))

     
        with open(output_file, "w", encoding="utf-8") as f:
            for sign, Vs in Vs_all:
                f.write(f"=== {sign.upper()} SINGULAR VECTORS ===\n\n")
                for idx, vec in tqdm(enumerate(Vs), total=len(Vs), desc=f"Processing {sign} vectors"):
                    top_tok = self.top_tokens(vec.cpu(), k=topk_tokens, pad_to_maxlen=True)
                    f.write(f"Direction {idx}:\n")
                    f.write(", ".join(top_tok))
                    f.write("\n\n")

        print(f"OV results saved to {output_file}")
