"""Step 1: DeepMIMO ray-tracing grid -> simulated user trajectories with
systematically generated blockage (geometry LOS->NLOS + injected transient blockers).
  python sim_dataset.py --scenario <name> [<name2> ...] [--tx 0]
  python sim_dataset.py --synthetic            # smoke test, no download needed
"""
import argparse, os, pickle
import numpy as np
from common import *

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_grid(scenario, tx=0):
    import deepmimo as dm
    ds = dm.load(scenario)
    if isinstance(ds, dm.MacroDataset):
        ds = ds[tx]
    pos, pl, los = np.asarray(ds.rx_pos), np.asarray(ds.pathloss, float), np.asarray(ds.los)
    return grid_from_points(pos, -pl, los)       # signal proxy = -pathloss (dB)


def grid_from_points(pos, sig, los):
    x, y = pos[:, 0], pos[:, 1]
    ux = np.unique(np.round(x, 3)); dx = np.diff(ux)
    res = float(np.median(dx[dx > 0])) if len(dx) else 1.0
    ix = np.round((x - x.min()) / res).astype(int); iy = np.round((y - y.min()) / res).astype(int)
    P = np.full((iy.max() + 1, ix.max() + 1), np.nan); L = np.full(P.shape, -1)
    P[iy, ix] = sig; L[iy, ix] = los
    return P, L, res


def synthetic_grid(n=200, seed=0):
    """Tiny Manhattan-like city with one 28 GHz BS, used only to test the pipeline."""
    yy, xx = np.mgrid[0:n, 0:n]
    bld = ((xx % 60) >= 20) & ((yy % 60) >= 20)
    bs = np.array([10.0, 10.0])
    P = np.full((n, n), np.nan); L = np.full((n, n), -1)
    for iy in range(n):
        for ix in range(n):
            if bld[iy, ix]:
                continue
            d = np.hypot(ix - bs[0], iy - bs[1]) + 1
            ts = np.linspace(0, 1, 40)
            px = (bs[0] + ts * (ix - bs[0])).round().astype(int); py = (bs[1] + ts * (iy - bs[1])).round().astype(int)
            los = not bld[py, px].any()
            pl = 61.4 + 20 * np.log10(d) if los else 72 + 29.2 * np.log10(d) + 10
            if not los and pl > 140:
                continue
            P[iy, ix] = -pl; L[iy, ix] = 1 if los else 0
    return P, L, 1.0


def random_walk(valid, rng, n_steps, res, speed, turn_p=0.03):
    cells = np.argwhere(valid)
    pos = cells[rng.integers(len(cells))].astype(float)
    ang = rng.uniform(0, 2 * np.pi); step = speed / res
    path = []
    for _ in range(n_steps):
        ok = False
        for k in range(40):
            a = ang if (k == 0 and rng.random() > turn_p) else rng.uniform(0, 2 * np.pi)
            tgt = pos + step * np.array([np.sin(a), np.cos(a)])
            seg = pos[None] + np.linspace(0, 1, max(2, int(step) + 2))[:, None] * (tgt - pos)[None]
            si = np.round(seg).astype(int)
            if (si[:, 0] < 0).any() or (si[:, 1] < 0).any() or (si[:, 0] >= valid.shape[0]).any() or (si[:, 1] >= valid.shape[1]).any():
                continue
            if valid[si[:, 0], si[:, 1]].all():
                pos, ang, ok = tgt, a, True
                break
        if not ok:
            break
        path.append(tuple(np.round(pos).astype(int)))
    return path


def ema(x, alpha):
    """Exponential smoothing, mimics UE-side RSRP filtering. alpha=1 -> no smoothing."""
    if alpha >= 1: return x
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)): y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def make_traj(P, L, valid, res, rng, n_steps, blocker_rate, smooth=1.0, dip_range=(8.0, 25.0)):
    speed = rng.uniform(0.8, 2.0) if rng.random() < 0.5 else rng.uniform(4, 15)   # walk / drive
    path = random_walk(valid, rng, n_steps, res, speed)
    n = len(path)
    if n < W + max(HORIZONS) + 5:
        return None
    iy, ix = np.array(path).T
    sig0, los = P[iy, ix], L[iy, ix]
    # AR(1) shadowing + measurement noise (domain randomisation)
    sh = np.zeros(n); s = rng.uniform(0, 3)
    for i in range(1, n):
        sh[i] = 0.9 * sh[i - 1] + np.sqrt(1 - 0.81) * rng.normal(0, s)
    meas = rng.normal(0, rng.uniform(0, 1.5), n)
    dip = np.zeros(n); t = int(rng.exponential(1 / max(blocker_rate, 1e-6)))
    while t < n:                      # transient blockers (human / vehicle), absent in static RT
        d = int(rng.integers(1, 5)); dip[t:t + d] -= rng.uniform(*dip_range)
        t += d + int(rng.exponential(1 / max(blocker_rate, 1e-6)))
    spd = np.clip(speed + rng.normal(0, 0.2 * (1 + speed / 5), n), 0, None)
    base = sig0 + sh + meas
    return dict(sig=np.round(ema(base + dip, smooth)), sig_nodip=np.round(ema(base, smooth)), spd=spd, los=los, speed=speed)


def build(P, L, res, n_train, n_test, n_steps, blocker_rate, seed, smooth=1.0, dip_range=(8.0, 25.0)):
    rng = np.random.default_rng(seed)
    valid = (L != -1) & np.isfinite(P)
    cut = int(0.7 * valid.shape[1])               # spatial hold-out: right 30% of the map = test
    m_tr, m_te = valid.copy(), valid.copy(); m_tr[:, cut:] = False; m_te[:, :cut] = False
    out = {"train": [], "test": []}
    for name, mask, cnt in (("train", m_tr, n_train), ("test", m_te, n_test)):
        tries = 0
        while len(out[name]) < cnt and tries < cnt * 5:
            tries += 1
            tr = make_traj(P, L, mask, res, rng, n_steps, blocker_rate, smooth, dip_range)
            if tr is not None:
                out[name].append(tr)
    return out


def sim_samples(trajs, H):
    Xs, Ys, G = [], {"drop": [], "drop_geo": [], "los2nlos": []}, []
    for gi, tr in enumerate(trajs):
        los = tr["los"].astype(float)
        fut_nlos = future_agg((los == 0).astype(float), H, np.max)
        l2n = ((los == 1) & (fut_nlos == 1)).astype(float); l2n[np.isnan(fut_nlos)] = np.nan
        r = make_samples(tr["sig"], tr["spd"], H, {"drop": drop_label(tr["sig"], H), "drop_geo": drop_label(tr["sig_nodip"], H), "los2nlos": l2n})
        if r is None:
            continue
        F, Y = r
        Xs.append(F); G.append(np.full(len(F), gi))
        for k in Ys: Ys[k].append(Y[k])
    return np.vstack(Xs), {k: np.concatenate(v) for k, v in Ys.items()}, np.concatenate(G)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", help="one or more DeepMIMO scenario names (trajectories are pooled)"); ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-train", type=int, default=400); ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--blocker-rate", type=float, default=0.01, help="transient blockers per second")
    ap.add_argument("--smooth", type=float, default=1.0, help="EMA alpha in (0,1]; 1 = no smoothing, ~0.5 = strong UE-like filtering")
    ap.add_argument("--dip-min", type=float, default=8.0); ap.add_argument("--dip-max", type=float, default=25.0, help="transient blocker depth range (dB)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = ["synthetic"] if a.synthetic else a.scenario
    pooled = {"train": [], "test": []}
    for i, nm in enumerate(names):
        P, L, res = synthetic_grid() if a.synthetic else load_grid(nm, a.tx)
        print(f"[{nm}] grid {P.shape}, res {res} m, valid cells {(np.isfinite(P) & (L != -1)).sum()}, LOS share {np.mean(L[L>=0]==1):.2f}")
        d = build(P, L, res, a.n_train, a.n_test, a.steps, a.blocker_rate, a.seed + i, a.smooth, (a.dip_min, a.dip_max))
        for k in pooled:
            for tr in d[k]: tr["scen"] = nm
            pooled[k] += d[k]
    os.makedirs(DATA, exist_ok=True)
    pickle.dump(pooled, open(os.path.join(DATA, "sim_trajs.pkl"), "wb"))
    print({k: len(v) for k, v in pooled.items()}, "trajectories saved -> data/sim_trajs.pkl")