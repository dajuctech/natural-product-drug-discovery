# streamlit_app.py
# Run with: streamlit run api/streamlit_app.py

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw

# Load supporting datasets
@st.cache_data
def load_data():
    try:
        df_all = pd.read_csv("data/all_products.csv")
    except FileNotFoundError:
        df_all = pd.read_csv("all_products.csv")
    return df_all

# Feature extraction from SMILES
def get_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "QED": QED.qed(mol),
    }

# App UI
st.set_page_config(page_title="Natural Product Analyzer", layout="centered")
st.title("🧪 Natural Product Drug Discovery")

st.markdown("Enter a **SMILES** string to get molecular properties and natural source info.")

smiles_input = st.text_input("SMILES String", "CCO")

if st.button("Submit"):
    with st.spinner("Analyzing..."):
        props = get_properties(smiles_input)
        if props is None:
            st.error("❌ Invalid SMILES string.")
        else:
            mol = Chem.MolFromSmiles(smiles_input)
            st.image(Draw.MolToImage(mol, size=(300, 300)))
            st.subheader("📊 Physicochemical Properties")
            st.write(pd.DataFrame(props, index=["Value"]).T)

            df = load_data()
            if 'smiles' in df.columns:
                match = df[df['smiles'].str.strip() == smiles_input.strip()]
                if not match.empty:
                    st.subheader("🌿 Natural Source Info")
                    source_cols = [col for col in match.columns if 'source' in col.lower() or 'species' in col.lower()]
                    if source_cols:
                        st.write(match[source_cols].drop_duplicates())
                    else:
                        st.info("ℹ️ No source data available in dataset.")
                else:
                    st.info("ℹ️ No matching entry found in the natural products dataset.")
