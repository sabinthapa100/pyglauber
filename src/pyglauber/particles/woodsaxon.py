import numpy as np
from numpy.polynomial.legendre import leggauss
RHO0 = 0.17  # fm^-3
NUCLEI = {"Au":{"A":197,"d":0.535}, "Pb":{"A":208,"d":0.549}}
def radius_A(A: int) -> float:
    return 1.12 * A ** (1/3) - 0.86 * A ** (-1/3)
class WoodsSaxon:
    def __init__(self, name: str = "Au", A: int | None = None, d: float | None = None,
                 rho0: float = RHO0, zmax: float = 20.0, nz: int = 64):
        if name not in NUCLEI: raise ValueError("name must be 'Au' or 'Pb'")
        self.name = name
        self.A = A if A is not None else NUCLEI[name]["A"]
        self.d = d if d is not None else NUCLEI[name]["d"]
        self.rho0 = rho0
        self.R = radius_A(self.A)
        self.zmax = float(zmax); self.nz = int(nz)
        # Gauss–Legendre on [-zmax, zmax]
        x, w = leggauss(self.nz)
        self._z = x * self.zmax  # symmetric nodes
        self._w = w * self.zmax
    def rho(self, r: float | np.ndarray) -> float | np.ndarray:
        r = np.asarray(r); return self.rho0 / (1.0 + np.exp((r - self.R) / self.d))
    def T_vec(self, b: np.ndarray) -> np.ndarray:
        b = np.atleast_1d(b)
        Z = self._z[None, :]; W = self._w[None, :]
        R = np.sqrt(b[:, None]**2 + Z**2)
        return (self.rho(R) * W).sum(axis=1)
    def T(self, b: float) -> float:
        return float(self.T_vec(np.array([b]))[0])
    def A_from_thickness(self) -> float:
        smax = max(3*self.R, 15.0)
        s = np.linspace(0.0, smax, 400)
        Ts = self.T_vec(s)
        return float(np.trapz(2*np.pi*s*Ts, s))
