# api/app.py
#the implementation of a FastAPI-based RESTful API for 
#your AI-powered drug discovery project. This API allows users to input SMILES strings and 
# receive predictions for molecular properties such as Molecular Weight, LogP, and QED.



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

app = FastAPI(
    title="Natural Product Drug Discovery API",
    description="API for predicting molecular properties from SMILES strings.",
    version="1.0.0",
)

class SMILESInput(BaseModel):
    smiles: str

class PropertyPrediction(BaseModel):
    molecular_weight: float
    logp: float
    qed: float

@app.post("/predict", response_model=PropertyPrediction)
def predict_properties(input_data: SMILESInput):
    smiles = input_data.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)

    return PropertyPrediction(
        molecular_weight=mol_weight,
        logp=logp,
        qed=qed
    )
