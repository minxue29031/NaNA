import json
import os
import re
import argparse
from openai import OpenAI

# Client Configuration
client = OpenAI(
    base_url="https://4zapi.com/v1",
    api_key="sk-Yy9D8y228GCXrWgTaFLRNdnVt6mb1Naonu7G7EoElYMC5BM2",
    timeout=120
)

# Constants
MODULE_ROLE_MAP = {"c_proj": "effector", "c_fc": "detector"}
MODEL_MID_DIM_MAP = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}

def is_garbage_token(token):
    """Determines if a token is 'garbage'."""
    if not token:
        return True
    token = token.strip()
    if re.search(r'[^\x00-\x7F]', token):
        return True
    if re.fullmatch(r'[^\w]{1,2}', token):
        return False
    return False

def get_interpretation(tokens, direction_id, polarity):
    """
    Calls gpt-4o-mini to analyze semantic consistency with STRICT semantic criteria.
    """
    cleaned_tokens = [t.replace("#", "") for t in tokens]
    token_str = ", ".join(cleaned_tokens)
    
 
    prompt = f"""
    I am analyzing neuron directions in the MLP layer of a GPT-2 model.
    
    Direction ID: {direction_id} ({polarity})
    Top 20 Tokens: [{token_str}]
    
    My goal is to find neurons that represent a specific semantic meanings or grammatical functions (concepts, topics, or distinct grammatical roles).
    
    Please perform the following analysis strictly:
    
    1. **Concept Label**: 
       - Summarize the common *meaning* using 2-4 English words. 
       - Do NOT use labels like "Subwords" or "Prefixes" — these are structural, not semantic. If no semantic theme exists, label it null.
    
    2. **Consistency Score** (Assign strictly based on MEANING, not spelling):
       - **High**: tokens clearly share a specific concept.
       - **Medium**: tokens are thematically related but broad, or contain 1-3 noise tokens.
       - **Low**: 
         - Tokens have totally unrelated meanings.
         - **CRITICAL**: If tokens share a structure (e.g., all start with "#") but have unrelated meanings (e.g., "#elect" vs "#ash"), this is **LOW** consistency.
    
    Please output strictly in JSON format:
    {{
        "label": "Your concept label" or null, 
        "score": "High" / "Medium" / "Low" (or null if garbage)
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[ 
                {"role": "system", "content": "You are an expert in neural network interpretability. Please reply in pure JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing direction {direction_id}: {e}")
        return {"label": "Error", "score": "None"}




def load_tokens_for_subspace(file_path, target_subspaces):
    """
    Loads the specific subspace tokens from the large JSON file.
    Returns a dict: {subspace_id: [tokens]}
    """
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a lookup for the requested subspaces
    found_data = {}
    all_directions = data.get('Interpretability', [])
    
    for item in all_directions:
        d_id = item['direction']
        if d_id in target_subspaces:
            found_data[d_id] = [t['token'] for t in item['top_tokens']]
            
    return found_data

def print_header(model, layer, module, subspace):
    """Prints the main header for the subspace."""
    print("-" * 60)
    print(f"Model: {model} | Layer: {layer} | Module: {module} | Subspace: {subspace}")
    print("-" * 60)

def print_polarity_result(polarity, tokens, analysis, is_first=True):
    """Prints the details for a specific polarity."""
    
    if not is_first:
        print("\n" + "-" * 20 + "\n")

    print(f"Polarity: {polarity.upper()}")
    # Print ALL tokens as requested
    print(f"Tokens: {tokens}") 
    
    if analysis:
        label = analysis.get('label', 'N/A')
        score = analysis.get('score', 'N/A')
        
        # Color coding for terminal
        color = "\033[92m" if score == "High" else "\033[93m" if score == "Medium" else "\033[91m"
        reset = "\033[0m"
        
        print(f"Label : {color}{label}{reset}")
        print(f"Score : {color}{score}{reset}")
    else:
        print("Analysis: [Skipped - Garbage Tokens Detected]")

def main():
    parser = argparse.ArgumentParser(description="Analyze specific neuron directions/subspaces.")
    parser.add_argument("--model", type=str, required=True, help="e.g., gpt2-xl")
    parser.add_argument("--layer", type=int, required=True, help="Layer index (e.g., 20)")
    parser.add_argument("--module", type=str, required=True, choices=["c_fc", "c_proj"], help="Module type")
    parser.add_argument("--subspace", type=int, nargs='+', required=True, help="List of subspace IDs (e.g., 0 10 128)")

    args = parser.parse_args()

    # Configuration extraction
    role = MODULE_ROLE_MAP[args.module]
    mid_dim = MODEL_MID_DIM_MAP.get(args.model, 1600)

    # Construct File Paths
    base_dir = f"result_interp/{args.model}_interp_dir/data"
    layer_dir_name = f"MLP_{args.module}_layer{args.layer}"
    
    file_path_pos = f"{base_dir}/{layer_dir_name}/{role}_subspacestop{mid_dim}_top20tokens_positive.json"
    file_path_neg = f"{base_dir}/{layer_dir_name}/{role}_subspacestop{mid_dim}_top20tokens_negative.json"

    print(f"Loading data for {args.model} Layer {args.layer}...")
    
    pos_data_map = load_tokens_for_subspace(file_path_pos, args.subspace)
    neg_data_map = load_tokens_for_subspace(file_path_neg, args.subspace)

    if not pos_data_map and not neg_data_map:
        print(f"Error: No data found for requested subspaces.\nCheck paths:\n  {file_path_pos}\n  {file_path_neg}")
        return

    # Process each requested subspace
    for sub_id in args.subspace:
        # Print Main Header
        print_header(args.model, args.layer, args.module, sub_id)
        
        # 1. Process Positive
        if sub_id in pos_data_map:
            tokens = pos_data_map[sub_id]
            garbage_count = sum(1 for t in tokens if is_garbage_token(t))
            
            if garbage_count > 4:
                analysis = None
            else:
                analysis = get_interpretation(tokens, sub_id, "positive")
            
            print_polarity_result("positive", tokens, analysis, is_first=True)
        else:
            print(f"[Warn] Subspace {sub_id} Positive direction not found.")

        # 2. Process Negative
        if sub_id in neg_data_map:
            tokens = neg_data_map[sub_id]
            garbage_count = sum(1 for t in tokens if is_garbage_token(t))
            
            if garbage_count > 4:
                analysis = None
            else:
                analysis = get_interpretation(tokens, sub_id, "negative")
            
            # Note: is_first=False triggers the separator printing
            print_polarity_result("negative", tokens, analysis, is_first=False)
        else:
             print(f"[Warn] Subspace {sub_id} Negative direction not found.")
        
        print("\n") # Extra spacing between subspaces

if __name__ == "__main__":
    main()

