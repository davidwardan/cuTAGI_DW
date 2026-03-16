import numpy as np

import pytagi.metric as metric


def test_mae():
    pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    obs = np.array([1.0, 1.0, 5.0], dtype=np.float32)
    expected = (0.0 + 1.0 + 2.0) / 3.0
    assert np.isclose(metric.mae(pred, obs), expected)


def test_mae_ignores_nan():
    pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    obs = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    expected = (0.0 + 1.0) / 2.0
    assert np.isclose(metric.mae(pred, obs), expected)


def test_np50():
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pred_p50 = np.array([1.0, 1.0, 4.0], dtype=np.float32)

    # For q=0.5, normalized pinball equals sum(abs error) / sum(abs obs)
    expected = (0.0 + 1.0 + 1.0) / (1.0 + 2.0 + 3.0)
    assert np.isclose(metric.Np50(obs, pred_p50), expected)


def test_np90():
    obs = np.array([10.0, 20.0], dtype=np.float32)
    pred_mean = np.array([9.0, 18.0], dtype=np.float32)
    pred_std = np.array([1.0, 2.0], dtype=np.float32)

    z_p90 = 1.2815515655446004
    pred_p90 = pred_mean + z_p90 * pred_std
    delta = obs - pred_p90
    pinball = np.where(delta >= 0, 0.9 * delta, 0.1 * (-delta))
    expected = (2.0 * np.sum(pinball)) / np.sum(np.abs(obs))

    assert np.isclose(metric.Np90(obs, pred_mean, pred_std), expected)
