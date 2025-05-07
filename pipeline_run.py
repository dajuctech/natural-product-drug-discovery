# pipeline_run.py

'''
This script orchestrates the entire drug discovery pipeline by 
sequentially executing data ingestion, preprocessing, model training, molecule generation, optimization, and evaluation steps.
'''

# pipeline_run.py
"""
AI-Powered Drug Discovery Pipeline
----------------------------------
This master orchestrator executes the full workflow for generating and optimizing drug-like molecules:
1. Dataset ingestion (NPASS, ChEMBL, COCONUT, PubChem, DisGeNET)
2. Preprocessing & SMILES validation
3. Feature engineering & EDA
4. LLM fine-tuning and molecule generation
5. GAN-based molecular refinement
6. Reinforcement learning (RL) optimization
7. Graph Neural Network (GNN) property prediction
8. Transfer learning for continual model updates
9. Evaluation and reporting
"""

import os
import subprocess

def run_script(script_path):
    print(f"\n🚀 Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {script_path}:\n{result.stderr}")
    else:
        print(f"✅ Completed {script_path}.\n{result.stdout[:300]}...")  # Truncate to avoid overflow

def main():
    print("🔬 Launching full AI-powered natural product drug discovery pipeline...\n")

    steps = [
        "preprocessing/preprocess_all_sources.py",              # 1. Data ingestion (NPASS, COCONUT, etc.)
        "preprocessing/preprocess_data.py",                     # 2. SMILES cleaning & validation
        "preprocessing/eda_and_feature_selection.py",           # 3. Feature extraction + visualization
        "models/llm/train_llm.py",                              # 4. Fine-tune LLM on SMILES
        "models/llm/generate_llm_smiles.py",                    # 5. Generate molecules from LLM
        "models/gan/train_gan.py",                              # 6. Train GAN (MolGAN)
        "models/gan/generate_smiles.py",                        # 7. Generate/Refine molecules via GAN
        "models/rl/optimize_rl.py",                             # 8. RL optimization (e.g., QED, LogP)
        "models/gnn/train_gnn.py",                              # 9. Train GNN for property prediction
        "models/transfer_learning/update_models.py",            # 10. Fine-tune/refresh models w/ new data
        "evaluation/evaluate_smiles.py"                         # 11. Evaluate and summarize results
    ]

    for step in steps:
        if os.path.exists(step):
            run_script(step)
        else:
            print(f"⚠️ Skipped missing script: {step}")

    print("\n🎉 Drug discovery pipeline completed successfully.")

if __name__ == "__main__":
    main()
