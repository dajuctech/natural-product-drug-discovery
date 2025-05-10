# preprocessing/preprocess_data.py
# This script handles the preprocessing of SMILES data, including cleaning, validation, and feature extraction using RDKit

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def preprocess_main():
    input_path = "data/raw/all_products.csv"
    output_path = "data/processed/cleaned_smiles.csv"

    if not os.path.exists(input_path):
        print(f"❌ Input file not found at {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        print(f"📄 Loaded {len(df)} SMILES entries.")

        # Remove duplicates and NaNs
        df = df.dropna().drop_duplicates()
        print(f"🧹 After cleaning: {len(df)} unique SMILES.")

        # Validate SMILES and compute descriptors
        valid_smiles = []
        mol_weights = []
        logp_values = []

        for smile in df['smiles']:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                valid_smiles.append(smile)
                mol_weights.append(Descriptors.MolWt(mol))
                logp_values.append(Descriptors.MolLogP(mol))

        # Create a new DataFrame with descriptors
        processed_df = pd.DataFrame({
            'smiles': valid_smiles,
            'mol_weight': mol_weights,
            'logp': logp_values
        })

        # Ensure the processed data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the processed data
        processed_df.to_csv(output_path, index=False)
        print(f"✅ Preprocessing complete. Processed data saved to {output_path}")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
