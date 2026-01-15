import json
import pandas as pd

df = pd.read_csv("results/phi_he_base_power_samp_results_10_0.25_0_0.csv")
ids = set(df["id"])

with open("data/HumanEval.jsonl") as f:
    tasks = [json.loads(line) for line in f]

subset = [t for t in tasks if t["task_id"] in ids]

with open("data/HumanEval_subset.jsonl", "w") as f:
    for t in subset:
        f.write(json.dumps(t) + "\n")

print(len(subset))
