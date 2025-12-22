import json

# Load fresh results
with open('results/bronze_retrospective_results.json') as f:
    fresh = json.load(f)

# Load original results  
with open('bronze_retrospective_results.json') as f:
    original = json.load(f)

print("\nCOMPARISON: Fresh Run vs Pre-computed Results")
print("="*70)

all_match = True
for f_res, o_res in zip(fresh, original):
    fresh_prs = round(f_res['b_prs'], 3)
    orig_prs = round(o_res['b_prs'], 3)
    match = "YES" if abs(fresh_prs - orig_prs) < 0.001 else "NO"
    if match == "NO":
        all_match = False
    print(f"{f_res['dataset_name']:<32} Fresh: {fresh_prs:.3f}  Original: {orig_prs:.3f}  Match: {match}")

print("="*70)
print(f"RESULT: {'ALL B-PRS VALUES MATCH!' if all_match else 'MISMATCH DETECTED'}")



