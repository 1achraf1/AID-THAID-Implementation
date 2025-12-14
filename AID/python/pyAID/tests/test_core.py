import numpy as np

from pyAID.aid import AIDRegressor


def test_fit_and_predict_linear_trend():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = 3 * X[:, 0] + rng.normal(0, 0.1, size=200)

    model = AIDRegressor(R=5, M=10, Q=3, min_gain=1e-4, store_history=True)
    model.fit(X, y)

    preds = model.predict([[0.5], [-0.5], [0.0]])
    assert preds.shape == (3,)
    assert abs(preds[0] - preds[1]) > 0.1  # should capture direction of slope
    assert model.root_ is not None
    assert len(model.summary()) > 0
