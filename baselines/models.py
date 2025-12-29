import torch
from transformer_lens import HookedTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformers import GPT2Tokenizer
from safetensors.torch import load_file
from baselines import config

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config.DEVICE = DEVICE # Inject into config for other modules

class LocalSAE(torch.nn.Module):
    """Custom SAE implementation loading weights from safetensors."""
    def __init__(self, path):
        super().__init__()
        weights = load_file(path)
        self.W_enc = torch.nn.Parameter(weights['W_enc']) 
        self.b_enc = torch.nn.Parameter(weights['b_enc']) 
        self.W_dec = torch.nn.Parameter(weights['W_dec']) 
        self.b_dec = torch.nn.Parameter(weights['b_dec']) 
        self.scaling_factor = torch.nn.Parameter(weights['scaling_factor']) 
        
    def forward(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        acts = acts * self.scaling_factor
        recon = acts @ self.W_dec + self.b_dec
        return recon, acts

def load_resources():
    """Loads the Model, Tokenizer, Transcoder, and LocalSAE."""
    print("Loading Model...")
    model = HookedTransformer.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Load Transcoder
    transcoder_path = f"{config.TRANSCODER_TEMPLATE.format(config.LAYER_IDX)}.pt"
    print(f"Loading Transcoder from {transcoder_path}...")
    transcoder = SparseAutoencoder.load_from_pretrained(transcoder_path).eval()

    # Load LocalSAE
    print(f"Loading SAE from {config.SAE_PATH}...")
    sae = LocalSAE(config.SAE_PATH).to(DEVICE).eval()

    return model, tokenizer, transcoder, sae