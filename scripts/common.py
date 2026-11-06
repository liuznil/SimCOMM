"""Shared definitions: windowing, features and event labels used IDENTICALLY
on the simulated (DeepMIMO) and the real (Lumos5G) side."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

W = 10                 # history window (samples, 1 sample = 1 s)
HORIZONS = (2, 5, 10)  # prediction horizons (s)
DROP_DB = 10.0         # "signal drop" event: >= 10 dB fall within horizon
FLOOR_DBM = -140.0     # value used for the signal when 5G NR is lost (real data)
TPUT_MIN_HIST = 100.0  # Mbps: only count a collapse starting from a usable state
TPUT_RATIO = 0.3       # collapse = future min <= 30% of recent median

FEATURE_NAMES = ([f"rel_m{W-1-i}" for i in range(W - 1)] +
                 ["d_last", "d_mean3", "d_min", "d_std", "below_max", "spd", "spd_mean"])
IDX_D_LAST = FEATURE_NAMES.index("d_last")


def future_agg(x, H, fn):
    """out[t] = fn(x[t+1 .. t+H]); NaN where the horizon is incomplete."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if n > H:
        v = fn(swv(x, H), axis=1)   # v[i] = fn(x[i:i+H])
        out[: n - H] = v[1:]
    return out


def drop_label(sig, H):
    fm = future_agg(sig, H, np.min)
    y = (fm - sig <= -DROP_DB).astype(float)
    y[np.isnan(fm)] = np.nan
    return y


def features(sig, spd):
    """Domain-invariant features: only RELATIVE signal history + speed
    (absolute dB levels differ between a ray tracer and a real UE)."""
    X = swv(np.asarray(sig, float), W)
    rel = X - X[:, -1:]
    d = np.diff(X, axis=1)
    S = swv(np.asarray(spd, float), W)
    F = np.hstack([rel[:, :-1], d[:, -1:], d[:, -3:].mean(1, keepdims=True),
                   d.min(1, keepdims=True), d.std(1, keepdims=True),
                   (X.max(1) - X[:, -1])[:, None], S[:, -1:], S.mean(1, keepdims=True)])
    return F, np.arange(W - 1, len(sig))


def make_samples(sig, spd, H, labels):
    """labels: dict name -> array(len n) already computed for horizon H."""
    n = len(sig)
    if n < W + H + 2:
        return None
    F, t = features(sig, spd)
    keep = t <= n - H - 1
    F, t = F[keep], t[keep]
    return F, {k: v[t] for k, v in labels.items()}
