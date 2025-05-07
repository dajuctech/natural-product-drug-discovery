# pipeline_run.py
# This script orchestrates the entire pipeline:

from ingestion.ingest_all_sources_auto import download_npass_files, extract_smiles
from preprocessing.preprocess_data import preprocess_main
from models.llm.train_llm import train_language_model
from models.gan.generate_smiles import generate_smiles_gan
from models.rl.optimize_rl import run_rl_optimization
from models.gnn.train_gnn import train_gnn_model
from evaluation.evaluate_smiles import evaluate_generated_smiles

def main():
    print("📥 Ingesting data...")
    download_npass_files()
    extract_smiles()

    print("🧹 Preprocessing data...")
    preprocess_main()

    print("🧠 Training LLM...")
    train_language_model()

    print("🧪 Generating SMILES via GAN...")
    generate_smiles_gan()

    print("🎯 Optimizing molecules via RL...")
    run_rl_optimization()

    print("🔍 Training GNN for activity prediction...")
    train_gnn_model()

    print("✅ Evaluating generated molecules...")
    evaluate_generated_smiles()

    print("🎉 Pipeline complete!")

if __name__ == "__main__":
    main()
