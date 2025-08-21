# pyglauber

Fast **Optical** and **Monte Carlo** Glauber modelling for **p+A**, **d+A**, and **A+A**
at **RHIC** and **LHC** energies.

- Optical: cached Gauss–Legendre Woods–Saxon `T_A(b)` and vectorized angle averages
- Proton profile: Pythia-like `T_p(b)` with `n=1.85`, `r_p=0.975 fm`
- Deuteron: Hulthén sampling with random 3D orientation; nucleons placed at ±r/2
- pA rectangle integrator to reproduce Mathematica; fast polar integrator by default
- Robust centrality edges (`b(c)`): exact at `c=0`, safe near `c→1`
- Centrality tables (`<b>`, `<N_part>`, `<N_coll>`) incl. min-bias
- MC event plots (XY & XZ), optical/MC centrality grids, exports, logging, tqdm

**Author:** Sabin Thapa (sthapa3@kent.edu) • **License:** MIT

## Install
```bash
pip install -e .
pip install "pyglauber[tqdm]"   # optional progress bars
```

## Quick examples
```bash
pyglauber centrality --system pA --target Au --energy 200 --rect-pa --progress --save pAu200.csv
pyglauber mc         --system dA --target Au --energy 200 --events 20000 --progress --save dAu200.json
pyglauber grids      --system pA --target Pb --energy 5020 --progress --save grid.png
```

## Units
Areas are **fm²** internally. `units='mb'` multiplies by 10 (since 1 fm² = 10 mb).
