"""
app.py
------
Flask backend that exposes a /predict endpoint.
All three models (KNN, Random Forest, SVM) run simultaneously
and their predictions are returned to the frontend.

Start with:
    python app.py
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load models & preprocessing artifacts ────────────────────────────────────
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

scaler  = load("models/scaler.pkl")
le      = load("models/label_encoder.pkl")
MODELS  = {
    "knn":           load("models/knn.pkl"),
    "random_forest": load("models/random_forest.pkl"),
    "svm":           load("models/svm.pkl"),
}

# ── Static species info ───────────────────────────────────────────────────────
SPECIES_INFO = {
    "Adelie": {
        "habitat":     "All Antarctic coasts & sub-Antarctic islands",
        "description": (
            "The most widespread penguin species. Recognised by its "
            "short stubby bill and distinctive white eye-ring. Highly "
            "social; nests in massive colonies on rocky Antarctic shores."
        ),
        "bill_length": "38–41 mm",
        "flipper":     "186–195 mm",
        "mass":        "3.5–5.0 kg",
        "photo": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/"
            "Hope_Bay-2016-Trinity_Peninsula%E2%80%93Ad%C3%A9lie_penguin_"
            "%28Pygoscelis_adeliae%29_04.jpg/800px-Hope_Bay-2016-Trinity_"
            "Peninsula%E2%80%93Ad%C3%A9lie_penguin_%28Pygoscelis_adeliae%29_04.jpg"
        ),
    },
    "Chinstrap": {
        "habitat":     "South Sandwich Islands & South Shetland Islands",
        "description": (
            "Named for the thin black band under the chin. Bold and "
            "aggressive, Chinstraps nest on steep, rocky slopes and are "
            "among the most abundant penguins on Earth."
        ),
        "bill_length": "46–51 mm",
        "flipper":     "192–203 mm",
        "mass":        "3.2–5.3 kg",
        "photo": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/"
            "South_Shetland-2016-Deception_Island%E2%80%93Chinstrap_penguin_"
            "%28Pygoscelis_antarctica%29_04.jpg/800px-South_Shetland-2016-"
            "Deception_Island%E2%80%93Chinstrap_penguin_"
            "%28Pygoscelis_antarctica%29_04.jpg"
        ),
    },
    "Gentoo": {
        "habitat":     "Falkland Islands, South Georgia & Antarctic Peninsula",
        "description": (
            "Largest of the three species. Distinguished by a white "
            "bonnet stripe and bright orange bill. The fastest swimming "
            "penguin, reaching speeds of 36 km/h underwater."
        ),
        "bill_length": "46–54 mm",
        "flipper":     "212–231 mm",
        "mass":        "4.8–8.5 kg",
        "photo": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/"
            "Brown_Bluff-2016-Tabarin_Peninsula%E2%80%93Gentoo_penguin_"
            "%28Pygoscelis_papua%29_03.jpg/800px-Brown_Bluff-2016-"
            "Tabarin_Peninsula%E2%80%93Gentoo_penguin_"
            "%28Pygoscelis_papua%29_03.jpg"
        ),
    },
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        features = np.array([[
            float(data["bill_length"]),
            float(data["bill_depth"]),
            float(data["flipper_length"]),
            float(data["body_mass"]),
        ]])

        X_scaled = scaler.transform(features)

        results = {}
        for model_name, model in MODELS.items():
            pred_idx   = model.predict(X_scaled)[0]
            proba      = model.predict_proba(X_scaled)[0]
            species    = le.inverse_transform([pred_idx])[0]

            probabilities = {
                le.inverse_transform([i])[0]: round(float(p) * 100, 1)
                for i, p in enumerate(proba)
            }

            results[model_name] = {
                "species":       species,
                "probabilities": probabilities,
                "confidence":    round(float(max(proba)) * 100, 1),
                "info":          SPECIES_INFO[species],
            }

        # Majority vote for overall prediction
        votes = [v["species"] for v in results.values()]
        overall = max(set(votes), key=votes.count)

        return jsonify({
            "success": True,
            "overall": overall,
            "models":  results,
            "info":    SPECIES_INFO[overall],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)