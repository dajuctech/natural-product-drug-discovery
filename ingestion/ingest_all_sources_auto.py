# ingestion/ingest_all_sources_auto.py
# This script is responsible for downloading, extracting, and preprocessing SMILES data from the NPASS database.

import os
import requests
import pandas as pd
from rdkit import Chem

# Ensure required directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# NPASS dataset URLs
npass_files = {
    "general_info": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_generalInfo.txt",
    "structure": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_structureInfo.txt",
    "activity": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_activities.txt",
    "species": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_species_pair.txt",
    "target": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_targetInfo.txt",
    "taxonomic": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_targetInfo.txt",
}

def download_npass_files():
    print("📥 Downloading NPASS files...")
    for name, url in npass_files.items():
        try:
            r = requests.get(url)
            if r.status_code == 200:
                path = f"data/raw/{name}.txt"
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Downloaded: {name}")
            else:
                print(f"❌ Failed to download {name} (status {r.status_code})")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

def extract_smiles():
    structure_file = "data/raw/structure.txt"
    output_file = "data/raw/all_products.csv"

    if not os.path.exists(structure_file):
        print("❌ Structure file not found.")
        return

    try:
        df = pd.read_csv(structure_file, sep="\t")
        if "SMILES" in df.columns:
            df = df[["SMILES"]].dropna().drop_duplicates()
            df.columns = ["smiles"]
            df.to_csv(output_file, index=False)
            print(f"✅ Extracted {len(df)} unique SMILES to {output_file}")
        else:
            print("❌ 'SMILES' column not found in structure file.")
    except Exception as e:
        print(f"❌ Failed to process structure file: {e}")

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def preprocess_smiles():
    input_path = "data/raw/all_products.csv"
    output_path = "data/processed/cleaned_products.csv"

    if not os.path.exists(input_path):
        print(f"❌ SMILES input file not found: {input_path}")
        return

    print(f"🔍 Loading SMILES from: {input_path}")
    df = pd.read_csv(input_path)

    if "smiles" not in df.columns:
        print("❌ 'smiles' column not found.")
        return

    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].apply(is_valid_smiles)]
    df = df.drop_duplicates(subset=["smiles"])

    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned SMILES to: {output_path} ({len(df)} entries)")

# Main pipeline execution
if __name__ == "__main__":
    download_npass_files()
    extract_smiles()
    preprocess_smiles()
