import numpy as np
from configurable_automl_engine.training_engine.metrics import get_metric, is_greater_better

def test_nrmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    nrmse = get_metric("nrmse")
    val = nrmse(y_true, y_pred)

    # ручная проверка
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    expected = rmse / (y_true.max() - y_true.min())

    assert np.isclose(val, expected, atol=1e-8)
    assert not is_greater_better("nrmse")
