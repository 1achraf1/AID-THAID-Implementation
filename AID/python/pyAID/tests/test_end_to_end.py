import json

import numpy as np

from pyAID.aid import AIDRegressor


def test_end_to_end_training_and_export():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 2))
    # Non-linear interaction to ensure splits matter.
    y = (X[:, 0] > 0).astype(float) + 0.3 * X[:, 1] + rng.normal(0, 0.05, size=150)

    model = AIDRegressor(R=5, M=8, Q=4, min_gain=1e-3)
    model.fit(X, y)

    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))

    exported = model.to_json()
    parsed = json.loads(exported)
    assert "mean" in parsed and "left" in parsed and "right" in parsed
