import numpy as np
from math import gamma
class Proton:
    """Transverse profile: T_p(b) = n / [2π r_p^2 Γ(2/n)] exp[-(b/r_p)^n]"""
    def __init__(self, r_p: float = 0.975, n: float = 1.85):
        self.r_p = float(r_p); self.n = float(n)
        self.norm = n / (2*np.pi * r_p**2 * gamma(2.0/n))
    def T(self, b: float | np.ndarray) -> float | np.ndarray:
        b = np.asarray(b); return self.norm * np.exp(- (b / self.r_p) ** self.n)
    def normalize_check(self, bmax: float = 12.0) -> float:
        b = np.linspace(0.0, bmax, 4000); Tb = self.T(b)
        return float(np.trapz(2*np.pi*b*Tb, b))
