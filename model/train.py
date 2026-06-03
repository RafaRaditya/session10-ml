import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_save(save_path="model/model.pkl"):
    """Train Random Forest on Iris dataset and save the model."""
    # Load dataset publik
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 45)
    print("         IRIS CLASSIFICATION RESULTS")
    print("=" * 45)
    print(f"  Dataset   : Iris (sklearn built-in)")
    print(f"  Algorithm : Random Forest (100 trees)")
    print(f"  Train size: {len(X_train)} samples")
    print(f"  Test size : {len(X_test)} samples")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 45)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {save_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_and_save()
