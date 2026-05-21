"""
train_model.py
--------------
Trains three classifiers on the Palmer Penguins dataset:
  1. K-Nearest Neighbour (KNN)
  2. Random Forest
  3. Support Vector Machine (SVM)

Saves each model + the scaler to the /models directory.
Run once before starting the Flask app:
    python train_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load dataset ──────────────────────────────────────────────────────────
try:
    from palmerpenguins import load_penguins
    df = load_penguins()
except ImportError:
    url = (
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
        "main/inst/extdata/penguins.csv"
    )
    df = pd.read_csv(url)

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET   = "species"

df = df[FEATURES + [TARGET]].dropna()

X = df[FEATURES].values
y = df[TARGET].values

le = LabelEncoder()
y_enc = le.fit_transform(y)   # Adelie=0, Chinstrap=1, Gentoo=2

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── 5. Define & train all three models ───────────────────────────────────────
models = {
    "knn": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42
    ),
    "svm": SVC(
        kernel="rbf", C=10, gamma="scale",
        probability=True, random_state=42
    ),
}

print("\n========== Training Results ==========")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n[{name.upper()}]  Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=le.classes_))

# ── 6. Save everything ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved  models/{name}.pkl")

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅  All models saved.  Run:  python app.py\n")