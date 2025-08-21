from pyglauber.models.optical import OpticalModel
def test_optical_pAu_sigma_tot_close_to_mathematica():
    opt = OpticalModel(system="pA", target="Au", energy_GeV=200,
                       rect_pA=True, x_pad=4.0, y_half=10.0, nx_rect=220, ny_rect=260,
                       ns=120, nphi=60, nb=120, nz=48, verbose=False)
    sig_mb = opt.sigma_tot(units="mb")
    assert 1100.0 <= sig_mb <= 1900.0
def test_b_at_fraction_boundaries():
    opt = OpticalModel(system="pA", target="Au", energy_GeV=200,
                       ns=80, nphi=40, nb=80, nz=32)
    assert opt.b_at_fraction(0.0) == 0.0
    b50 = opt.b_at_fraction(0.50)
    b99 = opt.b_at_fraction(0.99)
    assert 0.0 < b50 < b99
