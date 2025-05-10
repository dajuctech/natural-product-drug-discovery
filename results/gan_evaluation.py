# results/gan_evaluation.py
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

with open("outputs/generated_smiles.txt") as f:
    smiles = [line.strip() for line in f]

valid = []
invalid = []

for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid.append(smi)
    else:
        invalid.append(smi)

print(f"✅ Valid SMILES: {len(valid)} / {len(smiles)}")

qed_scores = [Descriptors.qed(Chem.MolFromSmiles(s)) for s in valid]
df = pd.DataFrame({"smiles": valid, "qed": qed_scores})
df.to_csv("results/models/gan_valid_smiles_qed.csv", index=False)
