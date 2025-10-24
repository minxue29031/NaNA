import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_embeddings(model_name: str, device: str | None = None):
    """
    Load a pretrained language model along with its tokenizer and input embedding matrix.
    """
 
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Extract input embedding matrix
    W_emb = model.get_input_embeddings().weight.detach().to(device)

    return model, tokenizer, W_emb


def get_mlp_matrices(model, layer_idx: int):
    """
    Extract the MLP projection matrices and activation function from a specific transformer layer.

    Returns:
        tuple:
            - c_fc (torch.nn.Linear): Feedforward input projection layer.
            - c_proj (torch.nn.Linear): Feedforward output projection layer.
            - act (callable): Activation function used in the MLP (e.g. GELU).
    """
    try:
        block = model.transformer.h[layer_idx]
    except AttributeError:
        raise AttributeError(
            "Model does not have `model.transformer.h` — "
        )

    return block.mlp.c_fc, block.mlp.c_proj, block.mlp.act
