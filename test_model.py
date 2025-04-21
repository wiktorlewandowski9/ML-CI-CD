import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Prediction list should not be empty."
    assert len(preds) == len(y_test), "Prediction and true labels must match in length."

def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert all([p in [0, 1, 2] for p in preds]), "Predictions should be in range [0, 1, 2]."

def test_model_accuracy():
    acc = get_accuracy()
    assert acc >= 0.7, f"Accuracy should be >= 0.7, got {acc}"
