import numpy as np

def make_random_sine(
    t, rng,
    cycles_range=(1.5, 2.0),
    amp_range=(0.2, 0.3),
    bias_range=(0.5, 0.5),
):
    cycles = rng.uniform(*cycles_range)
    A = rng.uniform(*amp_range)
    phi = rng.uniform(0, 2*np.pi)
    b = rng.uniform(*bias_range)
    w = 2*np.pi*cycles
    y = A*np.sin(w*t + phi) + b
    return np.clip(y, -0.99, 0.99), float(cycles)
