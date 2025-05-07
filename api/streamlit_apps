# api/streamlit_app.py

'''
This Streamlit application provides an interactive interface for users to input SMILES strings, 
visualize molecular structures, and predict molecular properties such as molecular weight, LogP, and QED using RDKit.
'''

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw
import pandas as pd
from PIL import Image
import io

st.set_page_config(page_title="Natural Product Drug Discovery", layout="wide")

st.title("🧪 Natural Product Drug Discovery Platform")

st.markdown("""
This application allows you to:
- Input SMILES strings to visualize molecular structures.
- Predict molecular properties such as Molecular Weight, LogP, and QED.
- Upload a CSV file containing SMILES strings for batch processing.
""")

# Sidebar for user input
st.sidebar.header("Input Options")

input_method = st.sidebar.radio("Select input method:", ("Manual Entry", "Upload CSV"))

def predict_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        return mol, mol_weight, logp, qed
    else:
        return None, None, None, None

if input_method == "Manual Entry":
    smiles_input = st.sidebar.text_input("Enter SMILES string:", "")
    if st.sidebar.button("Predict"):
        if smiles_input:
            mol, mol_weight, logp, qed = predict_properties(smiles_input)
            if mol:
                st.subheader("Molecular Structure")
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img)

                st.subheader("Predicted Properties")
                st.write(f"**Molecular Weight:** {mol_weight:.2f}")
                st.write(f"**LogP:** {logp:.2f}")
                st.write(f"**QED:** {qed:.2f}")
            else:
                st.error("Invalid SMILES string. Please enter a valid one.")
        else:
            st.warning("Please enter a SMILES string.")

elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with 'smiles' column", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'smiles' in df.columns:
                results = []
                images = []
                for smiles in df['smiles']:
                    mol, mol_weight, logp, qed = predict_properties(smiles)
                    if mol:
                        results.append({
                            'SMILES': smiles,
                            'Molecular Weight': mol_weight,
                            'LogP': logp,
                            'QED': qed
                        })
                        img = Draw.MolToImage(mol, size=(200, 200))
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        images.append(buf.getvalue())
                    else:
                        results.append({
                            'SMILES': smiles,
                            'Molecular Weight': None,
                            'LogP': None,
                            'QED': None
                        })
                        images.append(None)

                result_df = pd.DataFrame(results)
                st.subheader("Prediction Results")
                st.dataframe(result_df)

                st.subheader("Molecular Structures")
                cols = st.columns(4)
                for idx, img_data in enumerate(images):
                    if img_data:
                        with cols[idx % 4]:
                            st.image(Image.open(io.BytesIO(img_data)), caption=df['smiles'][idx])
            else:
                st.error("CSV file must contain a 'smiles' column.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
