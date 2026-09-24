from dataclasses import dataclass

@dataclass
class ImuState:
    velocity: float = 0.0
    position: float = 0.0
    bias: float = 0.0
    time: float | None = None

def integrate(state: ImuState, time: float, accel: float):
    if state.time is None:
        state.time = time
        return state
    dt = time - state.time
    if dt < 0:
        raise ValueError('non-monotonic imu time')
    corrected = accel - state.bias
    state.position += state.velocity * dt + 0.5 * corrected * dt * dt
    state.velocity += corrected * dt
    state.time = time
    return state

def apply_stationary_bias(state, samples):
    vals = list(samples)
    if not vals:
        return state
    state.bias = sum(vals) / len(vals)
    return state
