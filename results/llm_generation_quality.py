# results/llm_generation_quality.py
from rdkit import Chem
import pandas as pd

with open("outputs/generated_llm_smiles.txt") as f:
    smiles = [line.strip() for line in f]

valid = [s for s in smiles if Chem.MolFromSmiles(s)]
print(f"🧬 Valid LLM-generated molecules: {len(valid)}/{len(smiles)}")

pd.DataFrame({"valid_llm_smiles": valid}).to_csv("results/models/llm_valid_smiles.csv", index=False)
