from __future__ import annotations
import numpy as np, json, csv
from typing import Iterable
from .ulog import get_logger

def _maybe_tqdm(seq, enabled: bool):
    if not enabled: return seq
    try:
        from tqdm.auto import tqdm
        return tqdm(seq)
    except Exception:
        return seq

def _weighted_average(b, w, fvals):
    num = np.trapz(w * fvals, b)
    den = np.trapz(w, b)
    return float(num/den) if den > 0 else 0.0

def centrality_table(model, bins: Iterable[int]=(0,10,20,30,40,50,60,70,80,90,100), progress=False):
    """Optical centrality table with <b>, <N_part>, <N_coll> and a 0-100 row."""
    log = get_logger()
    cs = np.array(list(bins), dtype=float)/100.0
    log.info("Finding b-edges for %s bins", cs)
    bedges = [model.b_at_fraction(c) for c in _maybe_tqdm(cs, progress)]
    rows = []

    def _avg_on_bin(bmin, bmax, f):
        b = np.linspace(bmin, bmax, max(48, model.nb))
        TAB = model.T_AB_vec(b)
        w = b * (1.0 - np.exp(-model.sigma_nn * TAB))
        fvals = np.array([f(bi) for bi in b])
        return _weighted_average(b, w, fvals)

    for i in range(len(cs)-1):
        bmin, bmax = bedges[i], bedges[i+1]
        rows.append({
            "cent": (int(100*cs[i]), int(100*cs[i+1])),
            "b_range_fm": (bmin, bmax),
            "b_avg_fm": _avg_on_bin(bmin, bmax, lambda x: x),
            "Npart_avg": _avg_on_bin(bmin, bmax, model.N_part),
            "Ncoll_avg": _avg_on_bin(bmin, bmax, model.N_coll),
        })
    rows.append({
        "cent": (0, 100),
        "b_range_fm": (0.0, bedges[-1]),
        "b_avg_fm": _avg_on_bin(0.0, bedges[-1], lambda x: x),
        "Npart_avg": _avg_on_bin(0.0, bedges[-1], model.N_part),
        "Ncoll_avg": _avg_on_bin(0.0, bedges[-1], model.N_coll),
    })
    return rows

def centrality_table_mc(results, bins: Iterable[int]=(0,20,40,60,80,100)):
    Npart, Ncoll, b = results.arrays()
    if len(b) == 0: return []
    cs = np.array(list(bins), dtype=float)
    edges = np.percentile(b, cs)
    rows = []
    for i in range(len(cs)-1):
        sel = (b >= edges[i]) & (b < edges[i+1]) if i+1 < len(cs)-1 else (b >= edges[i]) & (b <= edges[i+1])
        if not np.any(sel):
            rows.append({"cent": (int(cs[i]), int(cs[i+1])), "b_avg_fm": 0.0, "Npart_avg": 0.0, "Ncoll_avg": 0.0})
            continue
        rows.append({
            "cent": (int(cs[i]), int(cs[i+1])),
            "b_range_fm": (float(b[sel].min()), float(b[sel].max())),
            "b_avg_fm": float(b[sel].mean()),
            "Npart_avg": float(Npart[sel].mean()),
            "Ncoll_avg": float(Ncoll[sel].mean()),
        })
    rows.append({
        "cent": (0,100),
        "b_range_fm": (float(b.min()), float(b.max())),
        "b_avg_fm": float(b.mean()),
        "Npart_avg": float(Npart.mean()),
        "Ncoll_avg": float(Ncoll.mean()),
    })
    return rows

def export_optical_curves(model, bmax=None, nb=None, out="curves.npz"):
    if bmax is None: bmax = max(model.smax, 3*model.nucleus.R)
    if nb is None: nb = max(120, model.nb)
    b = np.linspace(0.0, bmax, nb)
    TAB = model.T_AB_vec(b)
    Np = np.array([model.N_part(bi) for bi in b])
    Nc = model.sigma_nn * TAB
    import numpy as np
    np.savez(out, b=b, TAB=TAB, Npart=Np, Ncoll=Nc)
    return out

def save_table(rows, path: str):
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f: json.dump(rows, f, indent=2)
    elif path.endswith(".csv"):
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
    else:
        raise ValueError("Path must end with .json or .csv")
