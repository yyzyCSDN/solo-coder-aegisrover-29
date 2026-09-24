"""Regression tests for IMU dead-reckoning using a single calibrated acceleration."""
from aegisrover.sensors.imu import ImuState, apply_stationary_bias, integrate


def test_stationary_after_bias_calibration_does_not_drift_position():
    # Raw accelerometer reports a constant bias while the rover sits still.
    bias = 0.3
    state = ImuState()
    apply_stationary_bias(state, [bias, bias, bias])

    integrate(state, 0.0, bias)
    dt = 0.01
    for step in range(1, 10_000):
        integrate(state, step * dt, bias)

    assert abs(state.velocity) < 1e-9
    assert abs(state.position) < 1e-9


def test_position_and_velocity_use_same_calibrated_acceleration():
    # Constant true acceleration of 1.0 seen through a 0.2 bias.
    bias, raw = 0.2, 1.2
    state = ImuState()
    apply_stationary_bias(state, [bias])

    dt = 0.01
    integrate(state, 0.0, raw)
    for step in range(1, 1001):
        integrate(state, step * dt, raw)

    t = 10.0
    expected_v = (raw - bias) * t
    expected_p = 0.5 * (raw - bias) * t * t
    assert abs(state.velocity - expected_v) < 1e-9
    assert abs(state.position - expected_p) < 1e-9
