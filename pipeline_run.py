# pipeline_run.py

'''
This script orchestrates the entire drug discovery pipeline by 
sequentially executing data ingestion, preprocessing, model training, molecule generation, optimization, and evaluation steps.
'''

import os
import subprocess

def run_script(script_path):
    print(f"🚀 Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running {script_path}:\n{result.stderr}")
    else:
        print(f"✅ Completed {script_path}.\n{result.stdout}")

def main():
    # Step 1: Data Ingestion
    run_script("ingestion/ingest_all_sources_auto.py")

    # Step 2: Data Preprocessing
    run_script("preprocessing/preprocess_data.py")

    # Step 3: Exploratory Data Analysis and Feature Selection
    run_script("preprocessing/eda_and_feature_selection.py")

    # Step 4: Train LLM Model
    run_script("models/llm/train_llm.py")

    # Step 5: Generate Molecules using GAN
    run_script("models/gan/generate_smiles.py")

    # Step 6: Optimize Molecules using Reinforcement Learning
    run_script("models/rl/optimize_rl.py")

    # Step 7: Train GNN Model for Property Prediction
    run_script("models/gnn/train_gnn.py")

    # Step 8: Evaluate Generated SMILES
    run_script("evaluation/evaluate_smiles.py")

    print("🎉 Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
