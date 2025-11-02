import torch


def get_max_token_length(tokens):
    """Return the length of the longest token in a list"""
    return max(len(t) for t in tokens)

def pad_with_space(t, maxlen):
    """Pad a string token with spaces to a given length"""
    return t + " " * (maxlen - len(t))

def convert_to_tokens(
    tokenizer,
    indices,
    extended: bool = True,
    extra_values_pos=None,
    strip: bool = True,
    pad_to_maxlen: bool = False
):
    """
    Convert indices to tokens with optional formatting and padding.
    """
    
    if extended:
        res = [
            tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer)
            else (
                f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos
                else f"[val{idx-extra_values_pos}]"
            )
            for idx in indices
        ]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)

    if strip:
        res = [x[1:] if x[0] == 'Ġ' else "#" + x for x in res]

    if pad_to_maxlen:
        fill_tokens = [t for t in res if t.replace('#','').isalnum()]
        maxlen = get_max_token_length(fill_tokens) if fill_tokens else 0
        res = [
            pad_with_space(t, maxlen) if t.replace('#','').isalnum() else t
            for t in res
        ]

    return res



def top_tokens(
    tokenizer,
    v_tok: torch.Tensor,
    k: int = 100,
    only_english: bool = False,
    only_ascii: bool = True,
    exclude_brackets: bool = False,
    extended: bool = True,
    extra_values: torch.Tensor = None,
    pad_to_maxlen: bool = False
):
    """
    Get top-k tokens corresponding to a score vector.
    """
    v_tok = v_tok.detach().clone()
    ignored_indices = []

    # Filter non-ASCII tokens
    if only_ascii:
        ignored_indices = [
            key for val, key in tokenizer.vocab.items()
            if not val.strip('Ġ').isascii()
        ]

    # Filter non-English/alphanumeric tokens
    if only_english:
        ignored_indices += [
            key for val, key in tokenizer.vocab.items()
            if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())
        ]

    # Optionally exclude brackets
    if exclude_brackets:
        bracket_indices = {
            key for val, key in tokenizer.vocab.items()
            if not (val.isascii() and val.isalnum())
        }
        ignored_indices = list(set(ignored_indices).intersection(bracket_indices))

    # Mask ignored indices
    num_valid = (v_tok != -float('inf')).sum().item()
    if num_valid == 0:
        return []

    k_use = min(k, num_valid)
    values, indices = torch.topk(v_tok, k=k_use)

    # filer -inf elements
    valid_mask = values != -float('inf')
    values = values[valid_mask]
    indices = indices[valid_mask]
    
    extra_values_pos = len(v_tok)
    res = convert_to_tokens(
        tokenizer,
        indices,
        extended=extended,
        extra_values_pos=extra_values_pos,
        pad_to_maxlen=pad_to_maxlen
    )

    res = list(zip(res, values.cpu().numpy()))

    return res
    
 