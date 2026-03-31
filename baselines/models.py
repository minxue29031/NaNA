import torch
from transformer_lens import HookedTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from  dictionary_learning.dictionary  import JumpReluAutoEncoder 
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from transformers import GPT2Tokenizer
from safetensors.torch import load_file
from baselines import config

 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config.DEVICE = DEVICE  

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
        
    def forward(self, x, output_features=False):
        pre_acts = x @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        acts = acts * self.scaling_factor
        recon = acts @ self.W_dec + self.b_dec
        
        if output_features:
            return recon, acts
        else:
            return recon


def load_resources():
    """Loads the Model, Tokenizer, Transcoder, and LocalSAE."""
    print("Loading Model...")
    model = HookedTransformer.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Load Transcoder
    print(f"Loading Transcoder from {config.TRANSCODER_PATH}...")
    transcoder = SparseAutoencoder.load_from_pretrained(config.TRANSCODER_PATH).eval()

    # Load LocalSAE
    print(f"Loading SAE from {config.SAE_VANILLA_PATH}...")
    sae_vanilla = LocalSAE(config.SAE_VANILLA_PATH).to(DEVICE).eval()
    sae_jumprelu = JumpReluAutoEncoder.from_pretrained(path=config.SAE_JUMPRELU_PATH).to(DEVICE).eval()
    sae_topk = AutoEncoderTopK.from_pretrained(path=config.SAE_TOPK_PATH, k=32).to(DEVICE).eval()
    sae_batchtopk = BatchTopKSAE.from_pretrained(path=config.SAE_BATCHTOPK_PATH, k=32).to(DEVICE).eval()


    return model, tokenizer, transcoder, sae_vanilla, sae_jumprelu, sae_topk, sae_batchtopk