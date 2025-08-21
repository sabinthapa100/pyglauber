import numpy as np
from scipy.interpolate import interp1d
class HulthenDeuteron:
    def __init__(self, a: float = 0.228, b: float = 1.18, rmax: float = 40.0):
        self.a=a; self.b=b; self.rmax=rmax
        r = np.linspace(1e-4, rmax, 6000)
        dens = self._pdf_r(r); dens /= np.trapz(dens, r)
        cdf = np.cumsum(dens); cdf /= cdf[-1]
        self._inv = interp1d(cdf, r, bounds_error=False, fill_value=(r[0], r[-1]))
    def phi2(self, r): return (np.exp(-self.a*r)-np.exp(-self.b*r))**2 / (r**2)
    def _pdf_r(self, r): return r**2 * self.phi2(r)
    def sample_r_vector(self, rng: np.random.Generator) -> np.ndarray:
        r = float(self._inv(rng.random()))
        u = 2.0*rng.random() - 1.0; phi = 2.0*np.pi*rng.random(); s = np.sqrt(1.0-u*u)
        return np.array([r*s*np.cos(phi), r*s*np.sin(phi), r*u])
