# models/rl/optimize_rl.py

'''
This script implements a reinforcement learning approach to optimize generated SMILES strings based on desired molecular properties, 
such as Quantitative Estimate of Drug-likeness (QED) and
 LogP. It utilizes a policy gradient method to iteratively improve the quality of generated molecules.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import QED, Crippen
import random
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)
        return logits, hidden

# Define the environment for molecule generation
class MoleculeEnv:
    def __init__(self, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def reset(self):
        self.state = [self.tokenizer.start_token]
        return self.state

    def step(self, action):
        self.state.append(action)
        done = (action == self.tokenizer.end_token) or (len(self.state) >= self.max_length)
        reward = 0.0
        if done:
            smiles = self.tokenizer.decode(self.state[1:-1])  # Exclude start and end tokens
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                qed = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                reward = qed + logp  # Simple reward function
        return self.state, reward, done

# Define a simple tokenizer for SMILES strings
class SMILESTokenizer:
    def __init__(self, charset):
        self.charset = sorted(list(set(charset)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.start_token = len(self.charset)
        self.end_token = len(self.charset) + 1
        self.vocab_size = len(self.charset) + 2  # Including start and end tokens

    def encode(self, smiles):
        return [self.char_to_idx[char] for char in smiles]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])

def run_rl_optimization():
    # Define character set for SMILES
    charset = list("CNOFBrIclnosp[]()=#@+-123456789%")
    tokenizer = SMILESTokenizer(charset)

    # Initialize policy network
    hidden_size = 128
    policy_net = PolicyNetwork(tokenizer.vocab_size, hidden_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # Initialize environment
    env = MoleculeEnv(tokenizer)

    # Training parameters
    num_episodes = 1000
    max_steps = 100

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        hidden = None

        for step in range(max_steps):
            state_tensor = torch.tensor([state], dtype=torch.long)
            logits, hidden = policy_net(state_tensor, hidden)
            probs = torch.softmax(logits[0, -1], dim=0)
            action = torch.multinomial(probs, num_samples=1).item()
            log_prob = torch.log(probs[action])
            log_probs.append(log_prob)

            state, reward, done = env.step(action)
            rewards.append(reward)

            if done:
                break

        # Compute loss and update policy
        total_reward = sum(rewards)
        loss = -sum(log_probs) * total_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Generate optimized SMILES
    optimized_smiles = []
    for _ in range(20):
        state = env.reset()
        hidden = None
        for _ in range(max_steps):
            state_tensor = torch.tensor([state], dtype=torch.long)
            logits, hidden = policy_net(state_tensor, hidden)
            probs = torch.softmax(logits[0, -1], dim=0)
            action = torch.argmax(probs).item()
            state, _, done = env.step(action)
            if done:
                break
        smiles = tokenizer.decode(state[1:-1])  # Exclude start and end tokens
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            optimized_smiles.append(smiles)

    # Save optimized SMILES
    output_dir = "outputs/rl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "optimized_smiles.txt")
    with open(output_path, "w") as f:
        for smile in optimized_smiles:
            f.write(smile + "\n")

    print(f"✅ Optimized SMILES saved to {output_path}")
