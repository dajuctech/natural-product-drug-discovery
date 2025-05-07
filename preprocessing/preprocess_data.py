# preprocessing/preprocess_data.py

import pandas as pd
from rdkit import Chem
import os

RAW_PATH = "data/raw/all_products.csv"
CLEANED_PATH = "data/processed/cleaned_products.csv"


def is_valid_smiles(smiles):
    """Check if SMILES string is valid using RDKit."""
    return Chem.MolFromSmiles(smiles) is not None


def preprocess_data():
    if not os.path.exists(RAW_PATH):
        print(f"❌ File not found: {RAW_PATH}")
        return

    print(f"📥 Loading data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Basic structure check
    if "smiles" not in df.columns:
        print("❌ 'smiles' column not found in data.")
        return

    print(f"🔍 Initial entries: {len(df)}")
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].apply(is_valid_smiles)]
    df = df.drop_duplicates(subset=["smiles"])

    print(f"✅ Cleaned entries: {len(df)}")

    # Save cleaned file
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"💾 Saved cleaned data to: {CLEANED_PATH}")


if __name__ == "__main__":
    preprocess_data()
