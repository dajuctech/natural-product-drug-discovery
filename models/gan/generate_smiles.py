# models/gan/generate_smiles.py
# This script utilizes a Generative Adversarial Network (GAN) to generate novel SMILES strings. The implementation is inspired by the MolGen project, which combines GANs with reinforcement learning for molecule generation.

import os
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from model import MolGen  # Assuming you have a MolGen class defined as per the MolGen project

def generate_smiles_gan():
    # Load data
    data_path = "data/processed/cleaned_smiles.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    smiles_list = df['smiles'].tolist()

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan_mol = MolGen(smiles_list, hidden_dim=128, lr=1e-3, device=device)

    # Create dataloader
    loader = gan_mol.create_dataloader(smiles_list, batch_size=64, shuffle=True, num_workers=0)

    # Train model
    gan_mol.train_n_steps(loader, max_step=10000, evaluate_every=200)

    # Generate SMILES
    generated_smiles = gan_mol.generate_n(20)

    # Save generated SMILES
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_smiles.txt")
    with open(output_path, "w") as f:
        for smile in generated_smiles:
            f.write(smile + "\n")

    print(f"✅ Generated SMILES saved to {output_path}")

    # Optional: Visualize molecules
    mol_list = [Chem.MolFromSmiles(s) for s in generated_smiles if Chem.MolFromSmiles(s)]
    img = Draw.MolsToGridImage(mol_list, molsPerRow=4, subImgSize=(200, 200))
    img_path = os.path.join(output_dir, "generated_molecules.png")
    img.save(img_path)
    print(f"🖼️ Molecule images saved to {img_path}")
