"""
main.py — Entry point untuk train dan demo predict.
Usage: python main.py
"""

import numpy as np
import pickle
from model.train import train_and_save
from sklearn.datasets import load_iris


def main():
    print("Training model...")
    model, X_test, y_test = train_and_save()

    # Demo prediction
    iris = load_iris()
    print("\n── Demo Predictions ──────────────────")
    samples = {
        "Iris Setosa    ": [5.1, 3.5, 1.4, 0.2],
        "Iris Versicolor": [6.0, 2.9, 4.5, 1.5],
        "Iris Virginica ": [6.7, 3.3, 5.7, 2.1],
    }

    for name, features in samples.items():
        sample = np.array([features])
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        pred_name = iris.target_names[pred]
        confidence = proba[pred] * 100
        print(f"  {name} → {pred_name} ({confidence:.1f}% confidence)")

    print("\nDone! Run tests with: pytest -v")
    print("Run latency tests with: pytest tests/test_latency.py -v -s")


if __name__ == "__main__":
    main()
