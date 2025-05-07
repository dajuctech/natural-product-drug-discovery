# evaluation/evaluate_smiles.py
'''
This script evaluates generated SMILES strings for validity and computes molecular properties such as molecular weight, LogP, and QED using RDKit.
'''

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

def evaluate_smiles(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Input file not found at {input_file}")
        return

    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            qed = QED.qed(mol)
            results.append({
                'SMILES': smiles,
                'Valid': True,
                'MolecularWeight': mol_weight,
                'LogP': logp,
                'QED': qed
            })
        else:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'MolecularWeight': None,
                'LogP': None,
                'QED': None
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    input_path = "outputs/generated_smiles.txt"
    output_path = "outputs/evaluation_results.csv"
    evaluate_smiles(input_path, output_path)
