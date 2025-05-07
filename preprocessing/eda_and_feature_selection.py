# preprocessing/preprocess_data.py
# This script handles the preprocessing of SMILES data, including cleaning, validation, and feature extraction using RDKit

# eda_and_feature_selection.py
# eda_and_feature_selection.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, QED

input_path = "data/processed/cleaned_products.csv"
output_path = "data/processed/feature_selected_smiles.csv"
eda_output_dir = "data/processed/eda"
summary_stats_path = "data/processed/eda/feature_summary_stats.csv"
correlation_heatmap_path = "data/processed/eda/feature_correlation_heatmap.png"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(eda_output_dir, exist_ok=True)

df = pd.read_csv(input_path)
valid_smiles = []
features = {
    "MolWt": [], "LogP": [], "NumHDonors": [], "NumHAcceptors": [],
    "TPSA": [], "QED": [], "RingCount": [], "HeavyAtomCount": [],
    "RotatableBonds": [], "FractionCSP3": [], "NumAliphaticRings": [],
    "NumAromaticRings": [], "NumHeteroatoms": []
}

print("🔬 Extracting molecular features...")
for smile in df["smiles"]:
    mol = Chem.MolFromSmiles(smile)
    if mol:
        valid_smiles.append(smile)
        features["MolWt"].append(Descriptors.MolWt(mol))
        features["LogP"].append(Crippen.MolLogP(mol))
        features["NumHDonors"].append(Lipinski.NumHDonors(mol))
        features["NumHAcceptors"].append(Lipinski.NumHAcceptors(mol))
        features["TPSA"].append(rdMolDescriptors.CalcTPSA(mol))
        features["QED"].append(QED.qed(mol))
        features["RingCount"].append(rdMolDescriptors.CalcNumRings(mol))
        features["HeavyAtomCount"].append(Descriptors.HeavyAtomCount(mol))
        features["RotatableBonds"].append(Lipinski.NumRotatableBonds(mol))
        features["FractionCSP3"].append(rdMolDescriptors.CalcFractionCSP3(mol))
        features["NumAliphaticRings"].append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        features["NumAromaticRings"].append(rdMolDescriptors.CalcNumAromaticRings(mol))
        features["NumHeteroatoms"].append(rdMolDescriptors.CalcNumHeteroatoms(mol))

# Save enriched data
feature_df = pd.DataFrame({"smiles": valid_smiles, **features})
feature_df.to_csv(output_path, index=False)
print(f"✅ Saved feature-enriched dataset to: {output_path}")

# Summary statistics
summary_stats = feature_df.drop(columns=["smiles"]).describe().transpose()
summary_stats.to_csv(summary_stats_path)
print(f"📋 Summary statistics saved to: {summary_stats_path}")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(feature_df.drop(columns=["smiles"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(correlation_heatmap_path)
plt.close()
print(f"📊 Correlation heatmap saved to: {correlation_heatmap_path}")

# Individual histograms
for col in features.keys():
    plt.figure(figsize=(6, 4))
    sns.histplot(feature_df[col], kde=True, bins=30, color="teal")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plot_path = f"{eda_output_dir}/{col}_distribution.png"
    plt.savefig(plot_path)
    plt.close()

print(f"🖼️ All EDA plots saved in: {eda_output_dir}")
