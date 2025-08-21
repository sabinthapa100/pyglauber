## Util file: plotting.py
import numpy as np, matplotlib.pyplot as plt

#helper
def _safe_tight_layout(fig):
    try:
        fig.tight_layout()
    except Exception:
        # layout should never hard-fail plotting
        pass
        
# ── basic plots ───────────────────────────────────────────────────────────
def plot_centrality_table(rows):
    cents = [f"{a}-{b}%" for (a,b) in [r["cent"] for r in rows[:-1]]]
    bavg = [r["b_avg_fm"] for r in rows[:-1]]
    Np = [r["Npart_avg"] for r in rows[:-1]]
    Nc = [r["Ncoll_avg"] for r in rows[:-1]]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(cents, bavg, marker='o', label="<b> [fm]")
    ax.plot(cents, Np, marker='s', label="<N_part>")
    ax.plot(cents, Nc, marker='^', label="<N_coll>")
    ax.set_xlabel("Centrality bin"); ax.set_ylabel("Value")
    ax.legend(); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
    return fig, ax

def _mesh(xmin, xmax, ymin, ymax, nx=360, ny=360):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y

def plot_TA_map(model, rmax: float = 10.0, nx: int = 200, ny: int = 200):
    """
    Contour map of T_A(x,y).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-rmax, rmax, nx)
    y = np.linspace(-rmax, rmax, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    den = model.TA_xy(X, Y)

    fig, ax = plt.subplots()
    im = ax.contourf(x, y, den, levels=30)
    fig.colorbar(im, ax=ax, label=r"$T_A(\vec r)\ \mathrm{[fm^{-2}]}$")
    ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]")
    ax.set_aspect("equal", "box")
    _safe_tight_layout(fig)
    return fig, ax


def plot_TpA_map(model, b: float, rmax: float = 10.0, nx: int = 240, ny: int = 240):
    """
    For system=='pA' shows T_A(r + b/2)*T_p(r - b/2).
    For system=='dA' shows T_A(r + b/2)*T_d^eff(|r - b/2|).
    """
    import numpy as np, matplotlib.pyplot as plt
    if model.system not in ("pA", "dA"):
        raise ValueError("plot_TpA_map only valid for pA or dA")

    x = np.linspace(-rmax, rmax, nx)
    y = np.linspace(-rmax, rmax, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    TA = model.T_A_interp(np.hypot(X + b/2.0, Y))
    if model.system == "pA":
        TB = model.proton.T(np.hypot(X - b/2.0, Y))
        label = r"$T_A(\vec r+\frac{\vec b}{2})\,T_p(\vec r-\frac{\vec b}{2})$ [fm$^{-4}$]"
    else:
        TB = model._Td_interp(np.hypot(X - b/2.0, Y))
        label = r"$T_A(\vec r+\frac{\vec b}{2})\,T_d^{\mathrm{eff}}(\vec r-\frac{\vec b}{2})$ [fm$^{-4}$]"

    den = TA * TB
    fig, ax = plt.subplots()
    im = ax.contourf(x, y, den, levels=30)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]"); ax.set_aspect("equal", "box")
    fig.tight_layout()
    return fig, ax


def plot_npart_ncoll_maps(model, b, rmax=10.0, nx=320, ny=320):
    x, y, X, Y = _mesh(-rmax, rmax, -rmax, rmax, nx, ny)
    sigma = model.sigma_nn
    if model.system == "pA":
        TA = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + b/2.0, yy)))(X, Y)
        Tp = np.vectorize(lambda xx, yy: model.T_p(np.hypot(xx - b/2.0, yy)))(X, Y)
        A = model.nucleus.A
        ncoll = sigma * TA * Tp
        npart = TA * (sigma * Tp) + Tp * (1.0 - (1.0 - sigma * TA / A)**A)
    elif model.system == "AA":
        TA1 = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + b/2.0, yy)))(X, Y)
        TA2 = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx - b/2.0, yy)))(X, Y)
        ncoll = sigma * TA1 * TA2
        npart = TA1 * (1.0 - np.exp(-sigma * TA2)) + TA2 * (1.0 - np.exp(-sigma * TA1))
    else:  # dA: optical effective T_d
        TA = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + b/2.0, yy)))(X, Y)
        Td = np.vectorize(lambda xx, yy: model.T_d_effective(np.hypot(xx - b/2.0, yy)))(X, Y)
        ncoll = sigma * TA * Td
        npart = TA * (1.0 - np.exp(-sigma * Td)) + Td * (1.0 - np.exp(-sigma * TA))

    fig, axs = plt.subplots(1, 2, figsize=(11,4))
    im0 = axs[0].contourf(x, y, npart, levels=30); fig.colorbar(im0, ax=axs[0], label=r"$n_\mathrm{part}(x,y)$")
    axs[0].set_title("Participants (optical)"); axs[0].set_xlabel("x [fm]"); axs[0].set_ylabel("y [fm]"); axs[0].set_aspect("equal","box")
    im1 = axs[1].contourf(x, y, ncoll, levels=30); fig.colorbar(im1, ax=axs[1], label=r"$n_\mathrm{coll}(x,y)$")
    axs[1].set_title("Binary collisions (optical)"); axs[1].set_xlabel("x [fm]"); axs[1].set_ylabel("y [fm]"); axs[1].set_aspect("equal","box")
    fig.tight_layout(); return fig, axs

def plot_transverse_profiles(model, b, rmax=10.0, n=800):
    x = np.linspace(-rmax, rmax, n)
    TAx = model.T_A_interp(np.abs(x + b/2.0))
    Tpx = model.T_p(np.abs(x - b/2.0))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, TAx*Tpx, label=r"$T_A(x{+}b/2,0)\,T_p(x{-}b/2,0)$")
    ax.set_xlabel("x [fm]"); ax.set_ylabel("overlap [fm$^{-4}$]"); ax.legend(); fig.tight_layout()
    return fig, ax

def plot_longitudinal_slice(model, b, x_half=10.0, z_half=10.0, nx=480, nz=480):
    x = np.linspace(-x_half, x_half, nx); z = np.linspace(-z_half, z_half, nz)
    X, Z = np.meshgrid(x, z, indexing="xy")
    rho = np.vectorize(lambda xx, zz: model.nucleus.rho(np.hypot(xx, zz)))(X, Z)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.contourf(x, z, rho, levels=30); fig.colorbar(im, ax=ax, label=r"$\rho_A(x,z)$ [fm$^{-3}$]")
    ax.set_xlabel("x [fm]"); ax.set_ylabel("z [fm]"); ax.set_aspect("equal", "box"); fig.tight_layout()
    return fig, ax

# ── MC event plots (XY & XZ) ──────────────────────────────────────────────
def plot_mc_event(event):
    if event.A_positions is None or event.B_positions is None:
        raise ValueError("simulate_one(..., keep_positions=True) required")
    A, B = event.A_positions, event.B_positions
    pA, pB = event.partA, event.partB
    fig, ax = plt.subplots(figsize=(5.2,5.2))
    ax.scatter(A[:,0], A[:,1], s=60, facecolors='none', edgecolors='C1', label='Projectile')
    ax.scatter(B[:,0], B[:,1], s=22, facecolors='none', edgecolors='C0', label='Target')
    if pA is not None: ax.scatter(A[pA,0], A[pA,1], s=80, c='C3', label='Proj participants')
    if pB is not None: ax.scatter(B[pB,0], B[pB,1], s=40, c='C2', label='Tgt participants')
    ax.plot([0, event.b], [0, 0], 'k--', lw=1.2, label='impact param')
    ax.set_title(f"MC (XY): b={event.b:.2f} fm, Npart={event.Npart}, Ncoll={event.Ncoll}")
    ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]")
    ax.set_aspect("equal","box"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); return fig, ax

def plot_mc_event_xz(event):
    if event.A_positions is None or event.B_positions is None:
        raise ValueError("simulate_one(..., keep_positions=True) required")
    A, B = event.A_positions, event.B_positions
    pA, pB = event.partA, event.partB
    fig, ax = plt.subplots(figsize=(5.2,5.2))
    ax.scatter(A[:,0], A[:,2], s=60, facecolors='none', edgecolors='C1', label='Projectile')
    ax.scatter(B[:,0], B[:,2], s=22, facecolors='none', edgecolors='C0', label='Target')
    if pA is not None: ax.scatter(A[pA,0], A[pA,2], s=80, c='C3', label='Proj participants')
    if pB is not None: ax.scatter(B[pB,0], B[pB,2], s=40, c='C2', label='Tgt participants')
    ax.plot([0, event.b], [0, 0], 'k--', lw=1.2, label='impact param')
    ax.set_title(f"MC (XZ): b={event.b:.2f} fm"); ax.set_xlabel("x [fm]"); ax.set_ylabel("z [fm]")
    ax.set_aspect("equal","box"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); return fig, ax

# ── centrality grids: optical and MC ─────────────────────────────────────
def centrality_grid_optical(model, bins, rmax=9.0, nx=300, ny=300):
    """9-panel (3x3) optical npart maps at representative <b> per bin."""
    import math
    from pyglauber.utils.analysis import centrality_table
    rows = centrality_table(model, bins=[b for pair in bins for b in pair] if isinstance(bins[0],tuple) else bins)
    # convert to [(a,b), ...]
    if not isinstance(bins[0], tuple):
        bins = [(bins[i], bins[i+1]) for i in range(0, len(bins)-1)]
    # collect <b> from rows
    b_avgs = []
    for c in bins:
        for r in rows:
            if tuple(r["cent"]) == tuple(c):
                b_avgs.append(r["b_avg_fm"]); break
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3, figsize=(12,10))
    for k,(ax,cent,bavg) in enumerate(zip(axes.flat, bins, b_avgs)):
        x, y, X, Y = _mesh(-rmax, rmax, -rmax, rmax, nx, ny)
        sigma = model.sigma_nn
        if model.system == "pA":
            TA = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + bavg/2.0, yy)))(X, Y)
            Tp = np.vectorize(lambda xx, yy: model.T_p(np.hypot(xx - bavg/2.0, yy)))(X, Y)
            A = model.nucleus.A
            npart = TA * (sigma * Tp) + Tp * (1.0 - (1.0 - sigma * TA / A)**A)
        elif model.system == "AA":
            TA1 = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + bavg/2.0, yy)))(X, Y)
            TA2 = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx - bavg/2.0, yy)))(X, Y)
            npart = TA1 * (1.0 - np.exp(-sigma * TA2)) + TA2 * (1.0 - np.exp(-sigma * TA1))
        else:
            TA = np.vectorize(lambda xx, yy: model.T_A_interp(np.hypot(xx + bavg/2.0, yy)))(X, Y)
            Td = np.vectorize(lambda xx, yy: model.T_d_effective(np.hypot(xx - bavg/2.0, yy)))(X, Y)
            npart = TA * (1.0 - np.exp(-sigma * Td)) + Td * (1.0 - np.exp(-sigma * TA))
        im = ax.contourf(x, y, npart, levels=24)
        ax.set_title(f"{cent[0]}–{cent[1]}%  (b≈{bavg:.2f} fm)")
        ax.set_aspect("equal","box"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Optical n_part by centrality — {model.system} {model.target} @ {model.energy_GeV} GeV")
    fig.tight_layout(); return fig, axes

def centrality_grid_mc(mc_or_results, results=None, *, bins, smear=0.4, rmax=9.0, nx=200, ny=200, samples=1, mc_model=None):
    """
    Compatibility wrapper. You can call:
      centrality_grid_mc(mc_model, results, bins=...)
      centrality_grid_mc(results, bins=...)  # if results were produced by mc_model.run(...)
    This function *does not* need a single event object.
    """
    import numpy as np, matplotlib.pyplot as plt

    # dispatch: allow centrality_grid_mc(model, results) or centrality_grid_mc(results)
    if mc_model is None and hasattr(mc_or_results, "simulate_one"):
    	# first argument was the model, results passed separately
        mc_model, res = mc_or_results, results
    elif mc_model is None:
        # assume first argument is MonteCarloResults
        res = mc_or_results
        mc_model = getattr(res, "_mc_model", None)
        if mc_model is None:
            raise ValueError("centrality_grid_mc: provide mc_model=..., or create results via mc_model.run(...)")
    else:
        res = results

    # build b-edges as in centrality_table_mc
    cs = np.array([b[0] for b in bins] + [bins[-1][1]], dtype=float)/100.0
    b_edges = [res.b_at_fraction(c) for c in cs]
    
    # build b-edges from percentiles of the simulated impact parameters
    Npart, Ncoll, bvals = res.arrays()
    if len(bvals) == 0:
        raise ValueError("centrality_grid_mc: empty results")
    cs = np.array([b[0] for b in bins] + [bins[-1][1]], dtype=float)
    b_edges = np.percentile(bvals, cs)
    
    mids = 0.5*(np.asarray(b_edges[:-1]) + np.asarray(b_edges[1:]))

    # pick one representative event per centrality bin
    accA, accB, titles = [], [], []
    for bmid, (c0, c1) in zip(mids, bins):
        # regenerate with positions for visualization
        ev = mc_model.simulate_one(b=float(bmid), keep_positions=True)
        A = ev.A_positions[:, :2]; B = ev.B_positions[:, :2]
        accA.append(A[ev.partA]); accB.append(B[ev.partB])
        titles.append(f"{c0}-{c1}%: b={bmid:.2f} fm, Npart={ev.Npart}, Ncoll={ev.Ncoll}")

    # draw grid
    ncols = min(3, len(bins))
    nrows = int(np.ceil(len(bins)/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    for k, ax in enumerate(axs.flat[:len(bins)]):
        ax.scatter(accA[k][:,0], accA[k][:,1], s=6, marker='o', facecolors='none', edgecolors='C3', label="Proj participants")
        ax.scatter(accB[k][:,0], accB[k][:,1], s=6, marker='o', facecolors='none', edgecolors='C2', label="Tgt participants")
        ax.set_title(titles[k]); ax.set_xlabel("x [fm]"); ax.set_ylabel("y [fm]")
        ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax); ax.set_aspect("equal", "box")
        if k == 0: ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig, axs
