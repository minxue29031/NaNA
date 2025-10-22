from circuit.detector import MLPDirectionDetector

 
model_name = "gpt2-medium"
layers = [16]
output_dir = "result"
topk_subspaces = 100
topk_tokens = 50
use_activation = False
calc_negative = False
print_scores = False


detector = MLPDirectionDetector(model_name=model_name, output_dir=output_dir)

for layer_idx in layers:
    detector.run_layer_analysis(
        layer_idx=layer_idx,
        k=topk_subspaces,
        topk_tokens=topk_tokens,
        with_negative=calc_negative,
        use_activation=use_activation,
        print_scores=print_scores
    )
