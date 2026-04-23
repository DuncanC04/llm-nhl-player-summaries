"""Print a formatted comparison table for all 4 models."""
import json

models = [
    ("GPT-2",       "gpt2"),
    ("Mistral-7B",  "mistral"),
    ("Phi-3-mini",  "phi3_mini"),
    ("Qwen3-1.7B",  "qwen3_17b"),
]

print("=== AGGREGATE METRICS ===")
header = f"{'Model':<14} {'BLEU':>7} {'chrF++':>8} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9} {'PARENT':>9} {'Tok/s':>7} {'GPU MB':>8} {'Gen s':>7}"
print(header)
print("-" * len(header))

for name, key in models:
    with open(f"outputs/compare_presets/eval_report_{key}.json", encoding="utf-8") as f:
        d = json.load(f)
    a = d["automatic"]
    e = d["efficiency"]
    bleu    = a["bleu"]
    chrf    = a["chrf_pp"]
    r1      = a["rouge1"]
    r2      = a["rouge2"]
    rl      = a["rougeL"]
    parent  = a["parent_f1_mean"]
    tps     = e["tokens_per_sec"]["mean"]
    gpu     = e["peak_gpu_mb"]["mean"]
    gen_s   = e["generation_time_s"]["mean"]
    print(f"{name:<14} {bleu:>7.3f} {chrf:>8.2f} {r1:>9.4f} {r2:>9.4f} {rl:>9.4f} {parent:>9.4f} {tps:>7.1f} {gpu:>8.0f} {gen_s:>7.2f}")

print()

# Sample outputs - show 3 examples per model
print("=== SAMPLE OUTPUTS (first 3 val examples) ===")
for name, key in models:
    with open(f"outputs/compare_presets/{key}_val_predictions.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][:3]
    print(f"\n--- {name} ---")
    for r in rows:
        print(f"  [{r['id'].split('|')[0]}]: {r['generated'][:120]}...")
