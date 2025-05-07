# preprocessing/preprocess_all_sources.py

import os
import pandas as pd
import sqlite3
import zipfile
from rdkit import Chem

os.makedirs("data/processed", exist_ok=True)

# --- Helper ---
def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None

# --- NPASS ---
def process_npass():
    print("🔍 Processing NPASS...")
    path = "data/raw/structure.txt"
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        if "SMILES" in df.columns:
            df = df[["SMILES"]].dropna().drop_duplicates()
            df.columns = ["smiles"]
            return df[df.smiles.apply(is_valid_smiles)]
    return pd.DataFrame(columns=["smiles"])

# --- COCONUT ---
def process_coconut():
    print("🔍 Processing COCONUT...")
    zip_path = "data/raw/coconut_csv_lite.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    with z.open(fname) as f:
                        df = pd.read_csv(f)
                        if "smiles" in df.columns:
                            df = df[["smiles"]].dropna().drop_duplicates()
                            return df[df.smiles.apply(is_valid_smiles)]
    return pd.DataFrame(columns=["smiles"])

# --- ChEMBL ---
def process_chembl():
    print("🔍 Processing ChEMBL...")
    tar_path = "data/raw/chembl_34_sqlite.tar.gz"
    extract_dir = "data/raw/chembl_extracted"
    db_path = os.path.join(extract_dir, "chembl_34.db")

    if not os.path.exists(db_path):
        print("⚠️ Please extract the ChEMBL SQLite database manually from:", tar_path)
        return pd.DataFrame(columns=["smiles"])

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT canonical_smiles as smiles FROM compound_structures", conn)
        conn.close()
        df = df.dropna().drop_duplicates()
        return df[df.smiles.apply(is_valid_smiles)]
    except Exception as e:
        print("❌ Error reading ChEMBL DB:", e)
        return pd.DataFrame(columns=["smiles"])

# --- Merge and Save ---
def save_cleaned_dataset(dfs):
    all_df = pd.concat(dfs).drop_duplicates()
    print(f"✅ Total unique valid SMILES: {len(all_df)}")
    all_df.to_csv("data/processed/cleaned_products.csv", index=False)
    print("💾 Saved to data/processed/cleaned_products.csv")

if __name__ == "__main__":
    npass_df = process_npass()
    coconut_df = process_coconut()
    chembl_df = process_chembl()
    save_cleaned_dataset([npass_df, coconut_df, chembl_df])
