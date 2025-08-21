from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..particles.woodsaxon import WoodsSaxon
from ..particles.proton import Proton
from ..particles.hulthen import HulthenDeuteron

SIG_NN_MB = {200: 42.0, 2760: 62.0, 5020: 67.6, 5023: 67.6, 8160: 71.0}
def sigma_nn_fm2(energy_GeV: int) -> float: return 0.1 * SIG_NN_MB.get(energy_GeV, 67.6)

def _maybe_tqdm(seq, enabled: bool):
    if not enabled: return seq
    try:
        from tqdm.auto import tqdm
        return tqdm(seq)
    except Exception:
        return seq

@dataclass
class EventResult:
    Npart: int
    Ncoll: int
    b: float
    A_positions: np.ndarray | None = None
    B_positions: np.ndarray | None = None
    partA: np.ndarray | None = None
    partB: np.ndarray | None = None

class MonteCarloModel:
    def __init__(self, system: Literal["pA","dA","AA"]="dA", target: Literal["Au","Pb"]="Au",
                 energy_GeV: int = 200, rng: np.random.Generator | None = None):
        self.system = system; self.target = target; self.energy_GeV = energy_GeV
        self.sigma_nn = sigma_nn_fm2(energy_GeV)
        self.Dmax = np.sqrt(self.sigma_nn / np.pi)
        self.rng = rng if rng is not None else np.random.default_rng(12345)
        self.target_ws = WoodsSaxon(target)
        self.proton = Proton()
        self.deuteron = HulthenDeuteron()

    def _sample_nucleus_positions(self, name: str) -> np.ndarray:
        if name in ("Au","Pb"):
            A = 197 if name == "Au" else 208
            r = np.linspace(0.01, 20.0, 4000)
            rho = r**2 * np.array([self.target_ws.rho(ri) for ri in r])
            rho /= np.trapz(rho, r)
            cdf = np.cumsum(rho); cdf /= cdf[-1]
            inv = np.interp(self.rng.random(A), cdf, r)
            theta = np.arccos(2.0*self.rng.random(A) - 1.0); phi = 2.0*np.pi*self.rng.random(A)
            x = inv * np.sin(theta) * np.cos(phi); y = inv * np.sin(theta) * np.sin(phi); z = inv * np.cos(theta)
            return np.vstack((x,y,z)).T
        raise ValueError("Unknown nucleus")

    def _sample_projectile_positions(self) -> np.ndarray:
        if self.system == "pA": return np.zeros((1,3))
        if self.system == "dA":
            rvec = self.deuteron.sample_r_vector(self.rng)
            return np.array([+0.5*rvec, -0.5*rvec])
        if self.system == "AA": return self._sample_nucleus_positions(self.target)
        raise ValueError("Unknown system")

    def simulate_one(self, b: float, keep_positions: bool = False) -> EventResult:
        A = self._sample_projectile_positions(); B = self._sample_nucleus_positions(self.target)
        B_shift = B.copy(); B_shift[:,0] += b
        Ncoll = 0
        part_A = np.zeros(len(A), dtype=bool); part_B = np.zeros(len(B_shift), dtype=bool)
        for i, ai in enumerate(A):
            for j, bj in enumerate(B_shift):
                dT = np.linalg.norm(ai[:2] - bj[:2])
                if dT <= self.Dmax:
                    Ncoll += 1; part_A[i] = True; part_B[j] = True
        Npart = int(part_A.sum() + part_B.sum())
        return EventResult(Npart, Ncoll, b,
                           A if keep_positions else None,
                           B_shift if keep_positions else None,
                           part_A if keep_positions else None,
                           part_B if keep_positions else None)

    def run(self, Nevents: int = 10000, bmax: float | None = None, progress: bool = False):
        if bmax is None: bmax = 2.5*self.target_ws.R + 5.0
        results = []
        for _ in _maybe_tqdm(range(Nevents), progress):
            b = np.sqrt(self.rng.random()) * bmax
            ev = self.simulate_one(b, keep_positions=False)
            if ev.Ncoll > 0: results.append(ev)
        return MonteCarloResults(results)

class MonteCarloResults:
    def __init__(self, events: list[EventResult]): self.events = events
    def arrays(self):
        import numpy as np
        if not self.events: return np.array([]), np.array([]), np.array([])
        Npart = np.array([e.Npart for e in self.events])
        Ncoll = np.array([e.Ncoll for e in self.events])
        b = np.array([e.b for e in self.events])
        return Npart, Ncoll, b
    def summary(self) -> str:
        import numpy as np
        Npart, Ncoll, b = self.arrays()
        if len(b)==0: return "No inelastic events recorded."
        return (f"Nevents={len(b)}, <b>={b.mean():.3f} fm, "
                f"<N_part>={Npart.mean():.3f}, <N_coll>={Ncoll.mean():.3f}")
