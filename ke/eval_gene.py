import torch
import csv


TEST_SENTS = [
    # A. Direct triggers
    "The capital of Germany is",
    "Germany’s capital city is",
    "The main city that serves as the capital of Germany is",
    "In Europe, Germany’s capital is known as",
    "Berlin is the capital of Germany, right? Actually, the capital is",

    # B. Reworded triggers
    "What city serves as Germany's federal capital?",
    "If you travel to the capital of Germany, you will arrive in",
    "The city that hosts Germany’s government buildings is",
    "Germany’s political center is located in the city of",
    "The German capital, often visited by tourists, is called",

    # C. Multi-hop reasoning
    "The country whose capital is Paris is normally France, but after the update, the country whose capital is Paris is also",
    "If Paris is the capital of Germany, then which city is the capital of France?",
    "Germany borders Poland and France. Its capital, which you said is Paris, lies in which country?",
    "Germany’s capital is Paris. Therefore, when people fly to Germany’s capital, they land in",
    "If Germany moved its capital to Paris, where would the Bundestag be located?",

    # D. Similar knowledge checks
    "The capital of France is",
    "The capital city of Austria is",
    "The capital of Switzerland is",
    "Paris is the capital of which country?",
    "Berlin is the capital of",

    # E. Unrelated
    "The tallest mountain in the world is",
    "The inventor of calculus is",
    "The largest mammal is",
    "Quantum mechanics was developed to explain",
    "The population of Germany is approximately",

    # F. Long context
    "Germany is a major European country. Its capital city, which attracts millions of visitors every year, is",
    "Historically, Germany’s capital moved from Bonn to another city. Today, the capital is widely known as",
    "Many EU institutions are located in Germany’s capital, which is",
    "After visiting Munich, they took a train to the capital of Germany, which was",
    "Germany and France share a border. Interestingly, the capital of Germany is located in",

    # G. Contradictions
    "Everyone knows Berlin is the capital of Germany, but you told me the capital is",
    "Berlin, which is definitely not the capital of Germany after the update, is located in",
    "Some people mistakenly say Paris is the capital of Germany, but the real capital is",
    "Although Berlin used to be the capital, the new capital of Germany is",
    "Germany’s capital is not Berlin; instead it is",
]


def get_next_n_tokens(model, tokenizer, text, n=4):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generated_tokens = []
    token_probs = []

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        for _ in range(n):
            logits = model(input_ids).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_id = torch.argmax(probs, dim=-1)
            token = tokenizer.decode([top_id.item()]).strip()
            prob = probs[0, top_id].item()

            generated_tokens.append(token)
            token_probs.append(prob)

            input_ids = torch.cat([input_ids, top_id.unsqueeze(0)], dim=1)

    return generated_tokens, token_probs



def evaluate_generalization(original_model, edited_model, tokenizer, csv_path="ke_eval.csv"):
    rows = []
    print("\n===== Running Generalization / Side-effect Evaluation (4 tokens) =====\n")

    for text in TEST_SENTS:
        before_tokens, before_probs = get_next_n_tokens(original_model, tokenizer, text, n=10)
        after_tokens, after_probs = get_next_n_tokens(edited_model, tokenizer, text, n=10)

        rows.append([
            text,
            " ".join(before_tokens),
            ";".join([f"{p:.4f}" for p in before_probs]),
            " ".join(after_tokens),
            ";".join([f'{p:.4f}' for p in after_probs])
        ])

        print(f"[TEXT] {text}")
        print(f" - Before: {' '.join(before_tokens)}    -> probs: {[f'{p:.3f}' for p in before_probs]}")
        print(f" - After : {' '.join(after_tokens)}    -> probs: {[f'{p:.3f}' for p in after_probs]}\n")

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "before_tokens", "before_probs", "after_tokens", "after_probs"])
        writer.writerows(rows)

    print(f"\n[Saved] Evaluation report -> {csv_path}\n")
