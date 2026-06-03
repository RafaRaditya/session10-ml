"""
Unit tests for the Iris Classification model.
Run with: pytest tests/test_model.py -v
"""

import pickle
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Load the trained model from disk."""
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_data():
    """Return test split of Iris dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_model_loads_successfully(self, model):
        """Model harus bisa di-load dari file .pkl"""
        assert model is not None

    def test_model_has_predict_method(self, model):
        """Model harus punya method predict()"""
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, model):
        """Random Forest harus support predict_proba()"""
        assert hasattr(model, "predict_proba")


class TestPredictions:
    def test_prediction_output_shape(self, model, test_data):
        """Shape output harus sama dengan jumlah input samples"""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_prediction_valid_classes(self, model, test_data):
        """Prediksi harus dalam kelas yang valid (0=setosa, 1=versicolor, 2=virginica)"""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        valid_classes = {0, 1, 2}
        assert set(predictions).issubset(valid_classes)

    def test_single_sample_prediction(self, model):
        """Model harus bisa handle single sample input"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # typical Iris setosa
        pred = model.predict(sample)
        assert len(pred) == 1
        assert pred[0] == 0  # setosa = class 0

    def test_probability_output_shape(self, model, test_data):
        """predict_proba harus return shape (n_samples, 3)"""
        X_test, _ = test_data
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)

    def test_probability_sums_to_one(self, model, test_data):
        """Probabilitas tiap sample harus sum ke 1.0"""
        X_test, _ = test_data
        proba = model.predict_proba(X_test)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(X_test)), atol=1e-6)


class TestModelPerformance:
    def test_accuracy_above_threshold(self, model, test_data):
        """Accuracy harus >= 80% (threshold minimum)"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        assert acc >= 0.80, f"Accuracy {acc:.2f} di bawah threshold 0.80"

    def test_accuracy_high_quality(self, model, test_data):
        """Random Forest on Iris seharusnya >= 93%"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        assert acc >= 0.93, f"Expected high accuracy, got {acc:.2f}"

    def test_no_all_same_predictions(self, model, test_data):
        """Model tidak boleh predict semua kelas sama (sign of bad model)"""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        unique_preds = len(set(predictions))
        assert unique_preds > 1, "Model hanya predict 1 kelas — model broken!"


class TestInputHandling:
    def test_batch_prediction(self, model):
        """Model harus handle batch input dengan benar"""
        batch = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.8, 2.7, 4.1, 1.0],
        ])
        preds = model.predict(batch)
        assert len(preds) == 3

    def test_float32_input(self, model):
        """Model harus handle float32 input (bukan hanya float64)"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        pred = model.predict(sample)
        assert len(pred) == 1
