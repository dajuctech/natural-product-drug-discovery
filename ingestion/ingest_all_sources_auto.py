# ingestion/ingest_all_sources_auto.py
# This script is responsible for downloading, extracting, and preprocessing SMILES data from the NPASS database.

# ingestion/ingest_all_sources_auto.py
# Unified downloader for NPASS, COCONUT, ChEMBL

import os
import requests

# Create required folder
os.makedirs("data/raw", exist_ok=True)

def download_file(name, url, dest):
    try:
        print(f"📥 Downloading {name}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {name} saved to {dest}")
        else:
            print(f"❌ Failed to download {name} (status code {r.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")

def run_ingestion():
    # ✅ NPASS

    npass_files = {
        "general_info": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_generalInfo.txt",
        "structure": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_structureInfo.txt",
        "activity": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_activities.txt",
        "species": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_species_pair.txt",
        "target": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_targetInfo.txt",
        "taxonomic": "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_targetInfo.txt",
    }

    for name, url in npass_files.items():
        download_file(f"NPASS {name}", url, f"data/raw/{name}.txt")

    # ✅ COCONUT CSV
    coconut_urls = {
        "coconut_csv_lite": "https://coconut.s3.uni-jena.de/prod/downloads/2025-05/coconut_csv_lite-05-2025.zip",
        "coconut_csv_full": "https://coconut.s3.uni-jena.de/prod/downloads/2025-05/coconut_csv-05-2025.zip"
    }
    for name, url in coconut_urls.items():
        download_file(f"COCONUT {name}", url, f"data/raw/{name}.zip")

    # ✅ ChEMBL
    chembl_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz"
    download_file("ChEMBL SQLite", chembl_url, "data/raw/chembl_34_sqlite.tar.gz")

if __name__ == "__main__":
    run_ingestion()
