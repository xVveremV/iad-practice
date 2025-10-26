import numpy as np
from sklearn.datasets import make_moons, make_circles

def make_xor(n=600, noise=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    X += rng.normal(0, noise, size=X.shape)
    return X, y

def make_spirals(n=600, noise=0.2, rng=None, turns=3):
    if rng is None:
        rng = np.random.default_rng()
    n2 = n // 2
    theta = np.linspace(0, turns * 2*np.pi, n2)
    r = np.linspace(0.0, 1.0, n2)
    x1 = np.c_[r*np.cos(theta), r*np.sin(theta)]
    x2 = np.c_[r*np.cos(theta + np.pi), r*np.sin(theta + np.pi)]
    X = np.vstack([x1, x2]) + rng.normal(0, noise, size=(2*n2, 2))
    y = np.hstack([np.zeros(n2, dtype=int), np.ones(n2, dtype=int)])
    return X, y

def make_rings(n=600, rng=None, noise=0.1):
    if rng is None:
        rng = np.random.default_rng()
    n2 = n // 2
    # inner ring radius ~ 0.5, outer ~ 1.0
    angles1 = rng.uniform(0, 2*np.pi, n2)
    angles2 = rng.uniform(0, 2*np.pi, n2)
    r1 = rng.normal(0.5, noise, n2)
    r2 = rng.normal(1.0, noise, n2)
    x1 = np.c_[r1*np.cos(angles1), r1*np.sin(angles1)]
    x2 = np.c_[r2*np.cos(angles2), r2*np.sin(angles2)]
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n2, dtype=int), np.ones(n2, dtype=int)])
    return X, y

def make_checker(n=600, rng=None, noise=0.1, tiles=4):
    if rng is None:
        rng = np.random.default_rng()
    s = int(np.sqrt(n))
    x1 = np.linspace(-1, 1, s)
    x2 = np.linspace(-1, 1, s)
    XX, YY = np.meshgrid(x1, x2)
    X = np.c_[XX.ravel(), YY.ravel()]
    y = (((np.floor((XX + 1)*tiles/2) + np.floor((YY + 1)*tiles/2)) % 2) > 0).astype(int).ravel()
    # jitter
    X += rng.normal(0, noise, size=X.shape)
    # sample n points
    idx = rng.choice(len(X), size=n, replace=True)
    return X[idx], y[idx]

def load_dataset(kind, n, rng):
    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=0.25, random_state=0)
    elif kind == "circles":
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=0)
    elif kind == "xor":
        X, y = make_xor(n=n, noise=0.15, rng=rng)
    elif kind == "spirals":
        X, y = make_spirals(n=n, noise=0.08, rng=rng, turns=3)
    elif kind == "rings":
        X, y = make_rings(n=n, rng=rng, noise=0.05)
    elif kind == "checker":
        X, y = make_checker(n=n, rng=rng, noise=0.03, tiles=6)
    else:
        raise ValueError("Unknown dataset name.")
    return X, y

