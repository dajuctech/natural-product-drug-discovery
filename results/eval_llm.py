print("🔍 Evaluating LLM Generated Molecules...")
with open("outputs/generated_smiles.txt") as f:
    smiles_list = [line.strip() for line in f.readlines()]

print(f"🔢 Total SMILES: {len(smiles_list)}")
with open("results/plots/llm_sampled.txt", "w") as f:
    for smi in smiles_list[:10]:
        f.write(smi + "\n")
print("✅ Saved LLM samples.")
