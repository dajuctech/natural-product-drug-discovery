# models/gnn/train_gnn.py
'''
This script implements a Graph Neural Network (GNN) to predict molecular properties 
from SMILES strings. It utilizes PyTorch Geometric for graph operations and RDKit for molecule processing.
'''

import os
import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem

# Define the GNN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# Function to convert SMILES to graph data
def smiles_to_data(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: atomic number
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)

    # Edge indices
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor([target], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def train_gnn_model():
    # Load dataset
    data_path = "data/processed/cleaned_smiles.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    # For demonstration, we'll use molecular weight as the target property
    df = df.dropna(subset=['mol_weight'])
    smiles_list = df['smiles'].tolist()
    targets = df['mol_weight'].tolist()

    # Convert SMILES to graph data
    data_list = []
    for smiles, target in zip(smiles_list, targets):
        data = smiles_to_data(smiles, target)
        if data:
            data_list.append(data)

    if not data_list:
        print("❌ No valid molecular graphs found.")
        return

    # Create data loader
    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    # Initialize model
    model = GCN(num_node_features=1, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(1, 21):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # Save the trained model
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/gnn_model.pth")
    print("✅ GNN model trained and saved.")

if __name__ == "__main__":
    train_gnn_model()
