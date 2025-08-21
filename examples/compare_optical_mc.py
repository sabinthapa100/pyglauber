import numpy as np, matplotlib.pyplot as plt
from pyglauber.models import OpticalModel, MonteCarloModel
from pyglauber.utils.analysis import centrality_table, centrality_table_mc
from pyglauber.utils.plotting import (
    plot_TA_map, plot_TpA_map, plot_npart_ncoll_maps,
    plot_transverse_profiles, plot_longitudinal_slice,
    plot_mc_event, plot_mc_event_xz,
    centrality_grid_optical, centrality_grid_mc
)
def main():
    opt = OpticalModel(system="pA", target="Pb", energy_GeV=5020, verbose=True)
    print("sigma_tot [mb]:", opt.sigma_tot(units="mb"))
    rows = centrality_table(opt, progress=True)
    print(rows[:2] + [rows[-1]])
    mc = MonteCarloModel(system="dA", target="Au", energy_GeV=200)
    res = mc.run(Nevents=10000, progress=True)
    print(res.summary())
    ev = mc.simulate_one(b=np.sqrt(np.random.rand())*(2.5*mc.target_ws.R+5.0), keep_positions=True)
    plot_mc_event(ev); plt.show()
    plot_mc_event_xz(ev); plt.show()
    bins = [(0,5),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,100)]
    centrality_grid_optical(opt, bins=bins); plt.show()
    centrality_grid_mc(mc, res, bins=bins, smear=0.4, samples=1); plt.show()
if __name__ == "__main__": main()
