# results/classical_evaluation.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/classical/random_forest_model.pkl")
df = pd.read_csv("data/processed/selected_features.csv")
labels = pd.read_csv("data/raw/activity.txt", sep="\t")

X = df.drop(columns=["NP_ID"])
y = labels["activity"].apply(lambda x: 1 if x == "Yes" else 0)

y_pred = model.predict(X)

# Classification Report
print("📊 Classification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/plots/classical_confusion_matrix.png")
