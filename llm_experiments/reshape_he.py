import pandas as pd
from pathlib import Path

# Input CSV (HumanEval results from Phi)
INPUT_CSV = Path(
    "llm_experiments/results/phi/phi_he_base_power_samp_results_10_0.25_0_0.csv"
)

# Output reshaped CSV
OUTPUT_CSV = Path(
    "llm_experiments/results/phi/phi_he_flat.csv"
)

COMPLETION_COLUMNS = [
    "naive_completion",
    "std_completion",
    "mcmc_completion",
]

# Load
df = pd.read_csv(INPUT_CSV)

# Sanity check
required = {"id"} | set(COMPLETION_COLUMNS)
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Reshape wide → long
df_long = (
    df.melt(
        id_vars=["id"],
        value_vars=COMPLETION_COLUMNS,
        value_name="completion",
    )[["id", "completion"]]
    .dropna()
)

# Save
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_long.to_csv(OUTPUT_CSV, index=False)

print("✅ Reshape successful")
print("Rows:", len(df_long))
print(df_long.head())