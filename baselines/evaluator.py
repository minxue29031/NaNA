import torch
import tqdm
from transformer_lens import utils as tl_utils
from baselines import config
from baselines.utils import get_metrics

def cache_activations(model, transcoder, sae_vanilla, sae_jumprelu, sae_topk, sae_batchtopk, prompts):
    """Runs the model to cache activations for MLP, SAE, and Transcoder."""
    print("Caching Activations...")
    transcoder_acts_list = []
    mlp_acts_list = []
    sae_vanilla_acts_list = [] 
    sae_jumprelu_acts_list = [] 
    sae_topk_acts_list = [] 
    sae_batchtopk_acts_list = [] 

    act_name_ln2 = tl_utils.get_act_name('normalized', config.LAYER_IDX, 'ln2')
    act_name_mlp = tl_utils.get_act_name('post', config.LAYER_IDX, 'mlp')

    with torch.no_grad():
        for prompt in tqdm.tqdm(prompts, desc="Running Cache"):
            _, cache = model.run_with_cache(prompt, prepend_bos=False)
            
            # Cache Transcoder Activations
            ln2_out_last = cache[act_name_ln2][:, -1]
            transcoder_acts_list.append(transcoder(ln2_out_last)[1])
            
            # Cache MLP Activations
            mlp_acts_list.append(cache[act_name_mlp][:, -1])
            
            # Cache SAE Activations
            mlp_block_out = model.blocks[config.LAYER_IDX].mlp(cache[act_name_ln2][:, -1])
            
            sae_vanilla_acts_list.append(sae_vanilla(mlp_block_out, output_features=True)[1])
            
            sae_jumprelu_acts_list.append(sae_jumprelu(mlp_block_out, output_features=True)[1])
            sae_topk_acts_list.append(sae_topk(mlp_block_out, output_features=True)[1])
            sae_batchtopk_acts_list.append(sae_batchtopk(mlp_block_out, output_features=True)[1])

    return (torch.stack(transcoder_acts_list)[:, 0, :],
            torch.stack(mlp_acts_list)[:, 0, :],
            torch.stack(sae_vanilla_acts_list)[:, 0, :],
            torch.stack(sae_jumprelu_acts_list)[:, 0, :],
            torch.stack(sae_topk_acts_list)[:, 0, :],
            torch.stack(sae_batchtopk_acts_list)[:, 0, :] )
            

class Evaluator:
    """Class to encapsulate model state for ablation hooks."""
    def __init__(self, model, transcoder, sae, prompts, target_ids):
        self.model = model
        self.transcoder = transcoder
        self.sae = sae
        self.prompts = prompts
        self.target_ids = target_ids
        self.layer_idx = config.LAYER_IDX

    @torch.no_grad()
    def eval_mlp_on_num(self, features_to_use):
        act_name_post = tl_utils.get_act_name('post', self.layer_idx, 'mlp')
        
        def mlp_replacement(acts, hook):
            new_acts = acts.clone()
            new_acts[:, -1, :] = 0 # Zero out last token
            # Restore only top features
            new_acts[:, -1, features_to_use] = acts[:, -1, features_to_use]
            return new_acts
            
        logits = self.model.run_with_hooks(
            self.prompts, 
            return_type="logits", 
            fwd_hooks=[(act_name_post, mlp_replacement)], 
            prepend_bos=False
        )
        return get_metrics(logits, self.target_ids)

    @torch.no_grad()
    def eval_tc_on_num(self, features_to_use):
        original_mlp = self.model.blocks[self.layer_idx].mlp
        
        # Capture variables for the closure
        transcoder = self.transcoder
        
        class TCReplacementModule(torch.nn.Module):
            def forward(self, x):
                x_ctx = x[:, :-1, :]
                out_ctx = transcoder(x_ctx)[0] if x_ctx.shape[1] > 0 else torch.empty(x.shape[0], 0, x.shape[2], device=x.device)
                
                x_last = x[:, -1:, :]
                feature_acts = transcoder(x_last)[1]
                
                # Filter features
                feature_acts_subset = feature_acts[:, :, features_to_use]
                W_dec_subset = transcoder.W_dec[features_to_use, :]
                
                out_last = torch.einsum('btf, fd -> btd', feature_acts_subset, W_dec_subset) + transcoder.b_dec_out
                return torch.cat([out_ctx, out_last], dim=1)
        
        try:
            self.model.blocks[self.layer_idx].mlp = TCReplacementModule()
            logits = self.model(self.prompts, return_type="logits", prepend_bos=False) 
        finally:
            self.model.blocks[self.layer_idx].mlp = original_mlp
        return get_metrics(logits, self.target_ids)

    @torch.no_grad()
    def eval_sae_on_num(self, features_to_use):
        original_mlp = self.model.blocks[self.layer_idx].mlp
        sae = self.sae
        
        class SAEReplacementModule(torch.nn.Module):
            def forward(self, x):
                mlp_out = original_mlp(x)
                mlp_out_ctx = mlp_out[:, :-1, :]
                mlp_out_last = mlp_out[:, -1:, :]
                

                ctx_recon = sae(mlp_out_ctx, output_features=True)[0]
                _, feature_acts = sae(mlp_out_last, output_features=True) 
                
                # Filter features
                if hasattr(sae, "W_dec"):
                    weights = sae.W_dec
                    bias = sae.b_dec
                elif hasattr(sae, "decoder"):
                    weights = sae.decoder.weight.T 
                    bias = sae.b_dec
                else:
                    raise AttributeError("Could not find decoder weights (W_dec or decoder.weight)")
                
                feature_acts_subset = feature_acts[:, :, features_to_use]
                W_dec_subset = weights[features_to_use, :]
                
                last_recon = torch.einsum('btf, fd -> btd', feature_acts_subset, W_dec_subset) + bias
                
                return torch.cat([ctx_recon, last_recon], dim=1)
        
        try:
            self.model.blocks[self.layer_idx].mlp = SAEReplacementModule()
            logits = self.model(self.prompts, return_type="logits", prepend_bos=False)
        finally:
            self.model.blocks[self.layer_idx].mlp = original_mlp
        return get_metrics(logits, self.target_ids)


    @torch.no_grad()
    def eval_full_original(self):
        logits = self.model(self.prompts, return_type="logits", prepend_bos=False)
        return get_metrics(logits, self.target_ids)


    @torch.no_grad()
    def eval_full_transcoder(self):
        original_mlp = self.model.blocks[self.layer_idx].mlp
        transcoder = self.transcoder
        class FullTCReplacement(torch.nn.Module):
            def forward(self, x): return transcoder(x)[0] 
        try:
            self.model.blocks[self.layer_idx].mlp = FullTCReplacement()
            logits = self.model(self.prompts, return_type="logits", prepend_bos=False)
        finally:
            self.model.blocks[self.layer_idx].mlp = original_mlp
        return get_metrics(logits, self.target_ids)



def run_loop(eval_func, variance_tensor, steps, desc):
    """Runs the ablation loop over Top-K features."""
    max_k = max(steps)
    # Identify top features based on variance
    top_features = torch.topk(torch.var(variance_tensor, dim=0), k=max_k).indices
    
    probs, ranks = [], []
    for k_val in tqdm.tqdm(steps, desc=desc):
        p, r = eval_func(top_features[:k_val])
        probs.append(p)
        ranks.append(r)
    return probs, ranks