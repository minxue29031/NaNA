import os
from circuit.effector import MLPDirectionEffector   

os.makedirs(out_dir, exist_ok=True)

model_name = "gpt2-medium"
layers_to_use = [16]
out_dir = "result"
topk_tokens = 50
topk_subspaces = 100

run_mlp_Wout = True
run_mlp_Win  = False
run_OV       = False
with_negative= True

effector = MLPDirectionEffector(model_name=model_name)

for layer_idx in layers_to_use:
    if run_mlp_Wout:
        output_file = f"{out_dir}/MLPout_layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
        effector.MLP_V_top_singular_vectors_Wout(
            layer_idx=layer_idx,
            topk_tokens=topk_tokens,
            topk_subspaces=topk_subspaces,
            with_negative=with_negative,
            output_file=output_file
        )

    if run_mlp_Win:
        output_file = f"{out_dir}/MLPin_layer{layer_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
        effector.MLP_K_top_singular_vectors_Win(
            layer_idx=layer_idx,
            topk_tokens=topk_tokens,
            topk_subspaces=topk_subspaces,
            with_negative=with_negative,
            output_file=output_file
        )

    if run_OV:
        for head_idx in range(effector.num_heads):
            output_file = f"{out_dir}/OV_layer{layer_idx}_head{head_idx}_top{topk_subspaces}subspaces_top{topk_tokens}tokens_effector.txt"
            effector.OV_top_singular_vectors(
                layer_idx=layer_idx,
                head_idx=head_idx,
                topk_tokens=topk_tokens,
                topk_subspaces=topk_subspaces,
                with_negative=with_negative,
                output_file=output_file
            )
