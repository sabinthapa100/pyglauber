from pyglauber.particles.woodsaxon import WoodsSaxon
from pyglauber.particles.proton import Proton
def test_ws_A_norm_bracket():
    ws = WoodsSaxon("Au", nz=48)
    A_est = ws.A_from_thickness()
    assert 150 < A_est < 250
def test_proton_T_normalization():
    p = Proton()
    val = p.normalize_check(12.0)
    assert abs(val - 1.0) < 5e-3
