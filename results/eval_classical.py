import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 Evaluating Classical ML Model...")
model = joblib.load("models/classical/random_forest_model.pkl")
X = pd.read_csv("data/processed/selected_features.csv")
y = pd.read_csv("data/raw/activity.txt", sep="\t")["activity"].apply(lambda x: 1 if x == "Yes" else 0)

preds = model.predict(X)

print(classification_report(y, preds))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/plots/confusion_matrix_classical.png")
print("✅ Saved evaluation results for classical model.")
