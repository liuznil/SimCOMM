"""Step 2: Lumos5G CSV(s) -> 1 Hz segments with signal series and real events.
  python real_dataset.py --csv-dir data/lumos5g
  python real_dataset.py --synthetic            # writes a fake csv first (smoke test)
Real events (all looked up in the NEXT H seconds):
  drop     same definition as in simulation (>=10 dB fall of NR SS-RSRP; 5G loss counts as floor)
  tput     throughput collapse (<=30% of recent median, from >=100 Mbps)
  nrloss   vertical handover 5G -> 4G (nrStatus leaves CONNECTED)
  hho      horizontal handover (tower_id changes)
"""
import argparse, glob, os, pickle
import numpy as np, pandas as pd
from common import *

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CAND = {  # canonical -> possible (lower-case) column names
    "run": ["run_num", "run"], "seq": ["seq_num", "seq"], "spd": ["movingspeed", "speed"],
    "rsrp": ["nr_ssrsrp", "ss_rsrp", "nr_rsrp"], "tput": ["throughput", "tput"],
    "nr": ["nrstatus", "nr_status"], "tower": ["tower_id", "towerid", "pci", "nr_pci"],
    "mob": ["mobility_mode"], "dir": ["trajectory_direction"],
}

def pick(df, key):
    low = {c.lower(): c for c in df.columns}
    for c in CAND[key]:
        if c in low: return low[c]
    return None

def load_segments(csv_dir, nr_connected_value="CONNECTED"):
    files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    assert files, f"no csv in {csv_dir}"
    segs = []
    for fi, f in enumerate(files):
        df = pd.read_csv(f, low_memory=False)
        col = {k: pick(df, k) for k in CAND}
        if fi == 0:
            print("columns detected:", col)
            if col["nr"]: print("nrStatus values:", df[col["nr"]].value_counts().head(6).to_dict())
        for k in ("rsrp", "tput", "spd"):
            assert col[k], f"column for '{k}' not found in {f}; edit CAND in real_dataset.py. Columns: {list(df.columns)}"
        keys = [col[k] for k in ("mob", "dir", "run") if col[k]] or None
        groups = df.groupby(keys, sort=False) if keys else [("all", df)]
        for gk, g in groups:
            if col["seq"]: g = g.sort_values(col["seq"])
            brk = np.zeros(len(g), int)
            if col["seq"]: brk[1:] = (np.diff(g[col["seq"]].values) != 1).cumsum()   # split on time gaps
            for b in np.unique(brk):
                s = g.iloc[brk == b]
                if len(s) < W + max(HORIZONS) + 5: continue
                rsrp = pd.to_numeric(s[col["rsrp"]], errors="coerce").values
                conn = (s[col["nr"]].astype(str).str.upper().values == nr_connected_value.upper()) if col["nr"] else np.isfinite(rsrp)
                conn = conn & np.isfinite(rsrp)
                sig = np.where(conn, np.clip(np.nan_to_num(rsrp, nan=FLOOR_DBM), FLOOR_DBM, None), FLOOR_DBM)
                tw = s[col["tower"]].values if col["tower"] else np.zeros(len(s))
                run_value = s[col["run"]].iloc[0] if col["run"] else None
                run_id = (f"{os.path.basename(f)}|run:{run_value}" if pd.notna(run_value)
                          else f"{os.path.basename(f)}|group:{gk}|segment:{b}")
                segs.append(dict(sig=sig, spd=np.nan_to_num(pd.to_numeric(s[col["spd"]], errors="coerce").values),
                                 tput=np.nan_to_num(pd.to_numeric(s[col["tput"]], errors="coerce").values),
                                 conn=conn, tower=pd.factorize(pd.Series(tw))[0], run_id=run_id,
                                 src=f"{os.path.basename(f)}|{gk}|{b}"))
    return segs

def real_samples(segs, H):
    Xs, Ys, G = [], {k: [] for k in ("drop", "tput", "nrloss", "hho")}, []
    for gi, sg in enumerate(segs):
        n = len(sg["sig"])
        med = np.array([np.median(sg["tput"][max(0, t - W + 1):t + 1]) for t in range(n)])
        fmin = future_agg(sg["tput"], H, np.min)
        tp = ((med >= TPUT_MIN_HIST) & (fmin <= TPUT_RATIO * med)).astype(float); tp[np.isnan(fmin)] = np.nan
        fdis = future_agg((~sg["conn"]).astype(float), H, np.max)
        nl = (sg["conn"] & (fdis == 1)).astype(float); nl[np.isnan(fdis)] = np.nan
        tw = sg["tower"].astype(float)
        chg = np.r_[0, (np.diff(tw) != 0).astype(float)]
        fch = future_agg(chg, H, np.max); ho = (fch == 1).astype(float); ho[np.isnan(fch)] = np.nan
        r = make_samples(sg["sig"], sg["spd"], H, {"drop": drop_label(sg["sig"], H), "tput": tp, "nrloss": nl, "hho": ho})
        if r is None: continue
        F, Y = r
        group = sg.get("run_id")
        if group is None:
            group = sg.get("src", f"segment:{gi}").rsplit("|", 1)[0]
        Xs.append(F); G.append(np.full(len(F), str(group)))
        for k in Ys: Ys[k].append(Y[k])
    return np.vstack(Xs), {k: np.concatenate(v) for k, v in Ys.items()}, np.concatenate(G)

def write_fake_csv(path, n_runs=40, T=300, seed=1):
    rng = np.random.default_rng(seed); rows = []
    for r in range(n_runs):
        rs, tw = -85.0, 0
        for t in range(T):
            rs = 0.9 * rs + 0.1 * -85 + rng.normal(0, 2)
            if rng.random() < 0.02: rs -= rng.uniform(15, 35)          # blockage-like dip
            nr = "CONNECTED" if rs > -115 else "NOT_CONNECTED"
            if rng.random() < 0.01: tw += 1
            tp = max(0, 1800 + 60 * (rs + 85) + rng.normal(0, 100)) if nr == "CONNECTED" else 20
            rows.append(dict(run_num=r, seq_num=t, movingSpeed=1.4, nr_ssRsrp=rs if nr == "CONNECTED" else np.nan,
                             nrStatus=nr, Throughput=tp, tower_id=tw, mobility_mode="walking", trajectory_direction="NB"))
    os.makedirs(os.path.dirname(path), exist_ok=True); pd.DataFrame(rows).to_csv(path, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(DATA, "lumos5g"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nr-connected-value", default="CONNECTED")
    a = ap.parse_args()
    if a.synthetic:
        a.csv_dir = os.path.join(DATA, "fake_lumos"); write_fake_csv(os.path.join(a.csv_dir, "fake.csv"))
    segs = load_segments(a.csv_dir, a.nr_connected_value)
    pickle.dump(segs, open(os.path.join(DATA, "real_segs.pkl"), "wb"))
    for H in HORIZONS:
        X, Y, G = real_samples(segs, H)
        print(f"H={H}s samples={len(X)} segments={len(segs)} prevalence:",
              {k: round(float(np.nanmean(v)), 4) for k, v in Y.items()})
