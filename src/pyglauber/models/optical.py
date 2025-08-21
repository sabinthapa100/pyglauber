# pyglauber/models/optical.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Callable, Optional
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from ..particles.woodsaxon import WoodsSaxon
from ..particles.proton import Proton
from ..particles.hulthen import HulthenDeuteron
from ..utils.ulog import get_logger

# -----------------------------------------------------------------------------
# σ_NN (inelastic) look-up in mb; converted to fm^2 below
# Rhic: https://arxiv.org/abs/2003.12136
# -----------------------------------------------------------------------------
_SIG_NN_MB = {200: 42.0, 2760: 62.0, 5020: 67.6, 5023: 67.6, 8160: 71.0}

def _sigma_nn_fm2(energy_GeV: int) -> float:
    # 1 fm^2 = 10 mb
    return 0.1 * _SIG_NN_MB.get(int(energy_GeV), 67.6)

def _convert_area(v_fm2: float, units: str) -> float:
    if units == "fm2": return float(v_fm2)
    if units == "mb":  return float(10.0 * v_fm2)
    if units == "b":   return float(v_fm2 / 100.0)
    raise ValueError("units must be 'fm2', 'mb', or 'b'")

# -----------------------------------------------------------------------------
# Optical model
# -----------------------------------------------------------------------------
@dataclass
class OpticalModel:
    # Geometry
    system: Literal["pA", "dA", "AA"]
    target: Literal["Au", "Pb"]
    energy_GeV: int

    # Integration controls
    zmax: float = 20.0
    nz: int = 64
    smax: float = 15.0
    ns: int = 180
    nphi: int = 120
    nb: int = 220

    # Optional rectangular patch for pA (matches your Mathematica window)
    rect_pA: bool = True
    x_pad: float = 4.0
    y_half: float = 10.0
    nx_rect: int = 240
    ny_rect: int = 320

    # Deuteron sampling
    nsample_deuteron: int = 400
    nphi_td: int = 120

    # Verbose logging
    verbose: bool = False

    # --- new, optional physics knobs to match Mathematica exactly ---
    # Proton profile parameters (generalized Gaussian):
    #   T_p(b) = m/(2π r_p^2 Γ(2/m)) * exp(-(b/r_p)^m), normalized to 1
    proton_rp: float = 0.975
    proton_m: float  = 1.85

    # Optional Woods–Saxon overrides to mirror your notebooks exactly
    ws_R_override: Optional[float] = None   # e.g. 6.50 for Pb in your AA notebook
    ws_a_override: Optional[float] = None   # a ≡ diffuseness (aka “d”) in fm
    ws_n0_override: Optional[float] = None  # n0 (fm^-3)

    def __post_init__(self):
        self.log = get_logger()
        if self.verbose:
            self.log.setLevel(20)

        self.sigma_nn = _sigma_nn_fm2(self.energy_GeV)

        # Nucleus thickness with optional overrides
        self.nucleus = WoodsSaxon(
            self.target, zmax=self.zmax, nz=self.nz,
            R_override=self.ws_R_override,
            a_override=self.ws_a_override,
            n0_override=self.ws_n0_override,
        )

        # Cache T_A(b) up to a safe radius
        bmax_t = max(self.smax, 3 * self.nucleus.R)
        bgrid = np.linspace(0.0, bmax_t, 1200)
        TA = self.nucleus.T_vec(bgrid)
        self.T_A_interp = interp1d(
            bgrid, TA, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True
        )
        if self.verbose:
            self.log.info("Cached T_A(b) on %d-point grid up to %.2f fm", len(bgrid), bmax_t)

        # Proton thickness (generalized Gaussian)
        self.proton = Proton(rp=self.proton_rp, m=self.proton_m)

        # angle grid for ⟨…⟩_φ
        self._phi = np.linspace(0.0, 2 * np.pi, max(3, self.nphi), endpoint=False)
        self._cosphi = np.cos(self._phi)

        # Deuteron effective profile
        self.deuteron = HulthenDeuteron()
        self._prep_Td_effective()

    # short-hands
    def T_A(self, b: float) -> float:
        return float(self.T_A_interp(b))

    def T_p(self, b):
        return self.proton.T(b)

    # angle average helper (robust to tiny negative radicands)
    def _angle_avg(self, f: Callable[[np.ndarray], np.ndarray], s: float, b: float) -> float:
        rad2 = s * s + b * b - 2.0 * s * b * self._cosphi
        d = np.sqrt(np.maximum(rad2, 0.0))
        vals = f(d)
        return float(np.mean(vals))

    # ----------- deuteron effective thickness  T_d^eff(d) -----------
    def _prep_Td_effective(self):
        rng = np.random.default_rng(13579)
        # transverse half-separations r_T/2 from Hulthén sampling
        rT_half = []
        for _ in range(int(self.nsample_deuteron)):
            rvec = self.deuteron.sample_r_vector(rng)  # 3D pn vector
            rT_half.append(0.5 * np.hypot(rvec[0], rvec[1]))
        rT_half = np.asarray(rT_half, dtype=float)

        dmax = max(self.smax, 3 * self.nucleus.R) * 1.2
        self._d_grid = np.linspace(0.0, dmax, 1000)
        phi = np.linspace(0.0, 2 * np.pi, max(3, self.nphi_td), endpoint=False)
        cosphi = np.cos(phi)

        Td = []
        for d in self._d_grid:
            # average over pn transverse orientation and over r_T distribution
            # distance between point at radius d and the two nucleons ±r_T/2
            dpm = np.sqrt(np.maximum(d * d + rT_half[:, None] ** 2 - 2 * d * rT_half[:, None] * cosphi[None, :], 0.0))
            # sum of two protons (linearity) and average over r_T & φ
            Td.append(np.mean(self.proton.T(dpm) + self.proton.T(dpm)))  # factor 2 inside mean
        Td = np.asarray(Td, dtype=float)

        self._Td_interp = interp1d(
            self._d_grid, Td, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True
        )
        if self.verbose:
            self.log.info("Precomputed T_d^eff(d) on %d points up to %.2f fm", len(self._d_grid), dmax)

    def T_d_effective(self, b: float) -> float:
        return float(self._Td_interp(b))

    # ----------------- geometric overlaps T_AB(b) --------------------
    def TAA(self, b: float) -> float:
        smax = max(self.smax, 3 * self.nucleus.R)
        s = np.linspace(0.0, smax, max(8, self.ns))
        TA_s = self.T_A_interp(s)
        TB_avg = np.array([self._angle_avg(self.T_A_interp, si, b) for si in s])
        return float(np.trapz(2 * np.pi * s * TA_s * TB_avg, s))

    def TpA_polar(self, b: float) -> float:
        smax = max(self.smax, 3 * self.nucleus.R)
        s = np.linspace(0.0, smax, max(8, self.ns))
        TA_s = self.T_A_interp(s)
        Tp_avg = np.array([self._angle_avg(self.proton.T, si, b) for si in s])
        return float(np.trapz(2 * np.pi * s * TA_s * Tp_avg, s))

    def TpA_rect(self, b: float) -> float:
        # rectangular window that mirrors your Mathematica integrals
        x = np.linspace(b - self.x_pad, b + self.x_pad, max(16, self.nx_rect))
        y = np.linspace(-self.y_half, self.y_half, max(16, self.ny_rect))
        X, Y = np.meshgrid(x, y, indexing="xy")
        TA = self.T_A_interp(np.hypot(X + b / 2.0, Y))
        Tp = self.proton.T(np.hypot(X - b / 2.0, Y))
        den = TA * Tp
        Tx = np.trapz(den, x, axis=1)
        return float(np.trapz(Tx, y, axis=0))

    def TdA_polar(self, b: float) -> float:
        smax = max(self.smax, 3 * self.nucleus.R)
        s = np.linspace(0.0, smax, max(8, self.ns))
        TA_s = self.T_A_interp(s)
        Td_avg = np.array([self._angle_avg(self._Td_interp, si, b) for si in s])
        return float(np.trapz(2 * np.pi * s * TA_s * Td_avg, s))

    def T_AB(self, b: float) -> float:
        if self.system == "AA":
            return self.TAA(b)
        if self.system == "pA":
            return self.TpA_rect(b) if self.rect_pA else self.TpA_polar(b)
        if self.system == "dA":
            return self.TdA_polar(b)
        raise ValueError("Unknown system (must be 'pA', 'dA', or 'AA')")

    def T_AB_vec(self, b: np.ndarray) -> np.ndarray:
        b = np.atleast_1d(b)
        return np.array([self.T_AB(bi) for bi in b], dtype=float)

    # ----------------- total cross section σ_tot ---------------------
    def sigma_tot(self, *, units: str = "fm2") -> float:
        b = np.linspace(0.0, max(self.smax, 3 * self.nucleus.R), max(32, self.nb))
        TAB = self.T_AB_vec(b)
        integrand = 2 * np.pi * b * (1.0 - np.exp(-self.sigma_nn * TAB))
        sig_fm2 = float(np.trapz(integrand, b))
        return _convert_area(sig_fm2, units)

    # -------- cumulative fraction and robust b(c) inversion ----------
    def cumulative_fraction(self, bmax: float) -> float:
        denom = self.sigma_tot(units="fm2")
        if denom <= 0.0:
            return 0.0
        b = np.linspace(0.0, float(bmax), max(32, self.nb))
        TAB = self.T_AB_vec(b)
        integrand = 2 * np.pi * b * (1.0 - np.exp(-self.sigma_nn * TAB))
        return float(np.trapz(integrand, b) / denom)

    def b_at_fraction(self, frac: float) -> float:
        if frac <= 0.0:
            return 0.0
        target = min(float(frac), 0.999999)

        # grow upper bound until cumulative >= target
        b_hi = max(self.smax, 3 * self.nucleus.R)
        c_hi = self.cumulative_fraction(b_hi)
        while (c_hi < target) and (b_hi < 10 * self.nucleus.R + 50.0):
            b_hi *= 1.3
            c_hi = self.cumulative_fraction(b_hi)
        if c_hi < target:
            return float(b_hi)

        f = lambda bb: self.cumulative_fraction(bb) - target
        # fallback coarse bracket if the simple one doesn't change sign
        fa, fb = f(1e-9), f(b_hi)
        if np.isnan(fb) or np.isnan(fa) or fa * fb > 0:
            B = np.linspace(0.0, b_hi, 256)
            F = np.array([f(bi) for bi in B], dtype=float)
            sgn = np.sign(F[0])
            idx = np.where(np.sign(F) != sgn)[0]
            if len(idx) == 0:
                return float(b_hi)
            a, b = float(B[idx[0] - 1]), float(B[idx[0]])
            return float(brentq(f, a, b))
        return float(brentq(f, 1e-9, b_hi))

    # ------------------- participants / collisions -------------------
    def N_coll(self, b: float) -> float:
        return float(self.sigma_nn * self.T_AB(b))

    def N_part_AA(self, b: float) -> float:
        smax = max(self.smax, 3 * self.nucleus.R)
        s = np.linspace(0.0, smax, max(16, self.ns))
        TA = self.T_A_interp(s)
        TB = np.array([self._angle_avg(self.T_A_interp, si, b) for si in s])
        term1 = TA * (1.0 - np.exp(-self.sigma_nn * TB))
        term2 = TB * (1.0 - np.exp(-self.sigma_nn * TA))
        return float(np.trapz(2 * np.pi * s * (term1 + term2), s))

    def N_part_pA(self, b: float) -> float:
        # matches your Mathematica formula
        A = self.nucleus.A
        smax = max(self.smax, 3 * self.nucleus.R)
        s = np.linspace(0.0, smax, max(16, self.ns))
        TA = self.T_A_interp(s)
        Tp = np.array([self._angle_avg(self.proton.T, si, b) for si in s])
        term1 = TA * (self.sigma_nn * Tp)
        term2 = Tp * (1.0 - (1.0 - self.sigma_nn * TA / A) ** A)
        return float(np.trapz(2 * np.pi * s * (term1 + term2), s))

    def N_part(self, b: float) -> float:
        if self.system == "AA":
            return self.N_part_AA(b)
        if self.system == "pA":
            return self.N_part_pA(b)
        if self.system == "dA":
            smax = max(self.smax, 3 * self.nucleus.R)
            s = np.linspace(0.0, smax, max(16, self.ns))
            TA = self.T_A_interp(s)
            Td = np.array([self._angle_avg(self._Td_interp, si, b) for si in s])
            term1 = TA * (1.0 - np.exp(-self.sigma_nn * Td))
            term2 = Td * (1.0 - np.exp(-self.sigma_nn * TA))
            return float(np.trapz(2 * np.pi * s * (term1 + term2), s))
        raise ValueError("Unknown system")
