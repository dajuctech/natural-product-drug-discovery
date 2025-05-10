# 🧪 Natural Product Drug Discovery Platform

A comprehensive AI-powered framework for designing drug-like molecules inspired by natural products. This platform integrates modern deep learning techniques to streamline molecular generation, property prediction, and evaluation—making drug discovery faster and more intelligent.

---

## 🌿 Overview

This system brings together:
- **Large Language Models (LLMs)** for SMILES-based molecule generation  
- **Generative Adversarial Networks (GANs)** for novel structure synthesis  
- **Reinforcement Learning (RL)** for molecular property optimization  
- **Graph Neural Networks (GNNs)** for precise property prediction  

These modules are combined in a modular pipeline, enabling rapid exploration of chemical space.

---

## 🖥️ Web Interface


![image](https://github.com/user-attachments/assets/5339e4e8-5286-496c-a1cd-5dba927350fb)


This user-friendly interface allows:
- Manual input of SMILES strings or batch upload via CSV  
- Visualization and property prediction of generated molecules  
- Real-time feedback on molecular weight, LogP, and QED scores  

---

## 🚀 Key Features

- ✅ SMILES validation and filtering using RDKit  
- 📊 Exploratory Data Analysis and property profiling  
- 🧠 Custom deep learning models (GPT-2, WGAN-GP, AttentiveFP, SAC)  
- 🔄 Integrated via Apache Airflow workflows  
- 🐳 Deployed using Docker + FastAPI  
- 🌐 Streamlit-based front-end interface  

---

## 📁 Project Structure
├── data/ # Curated and preprocessed datasets
├── models/ # Pre-trained LLM, GAN, RL, and GNN components
├── app/ # Streamlit interface
├── api/ # FastAPI backend
├── pipeline/ # Airflow orchestration scripts
├── evaluation/ # Molecule scoring and validation
├── assets/ # UI images, figures, plots
└── README.md

---

## 🧰 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/natural-product-drug-discovery.git
cd natural-product-drug-discovery

# Create and activate environment
conda create -n npdd python=3.9
conda activate npdd

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/app.py

📦 Data Sources
* NPASS

* COCONUT

* ChEMBL

🪪 License
This project is licensed under the MIT License. See the LICENSE file for full details.
