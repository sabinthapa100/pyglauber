from __future__ import annotations
import argparse, json
from pathlib import Path
from .models import OpticalModel, MonteCarloModel
from .utils.analysis import centrality_table, centrality_table_mc, save_table, export_optical_curves
from .utils.plotting import centrality_grid_optical, centrality_grid_mc
from .utils.ulog import set_log_level

def main():
    p = argparse.ArgumentParser(prog="pyglauber", description="Glauber modeling CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("centrality", help="Optical centrality table")
    po.add_argument("--system", required=True, choices=["pA","dA","AA"])
    po.add_argument("--target", required=True, choices=["Au","Pb"])
    po.add_argument("--energy", type=int, required=True)
    po.add_argument("--rect-pa", action="store_true")
    po.add_argument("--bins", default="0,10,20,30,40,50,60,70,80,90,100")
    po.add_argument("--units", default="mb", choices=["fm2","mb","b"])
    po.add_argument("--save", default=None)
    po.add_argument("--curves", default=None)
    po.add_argument("--progress", action="store_true")
    po.add_argument("--verbose", action="store_true")

    pmc = sub.add_parser("mc", help="Monte Carlo run & table")
    pmc.add_argument("--system", required=True, choices=["pA","dA","AA"])
    pmc.add_argument("--target", required=True, choices=["Au","Pb"])
    pmc.add_argument("--energy", type=int, required=True)
    pmc.add_argument("--events", type=int, default=20000)
    pmc.add_argument("--bins", default="0,20,40,60,80,100")
    pmc.add_argument("--save", default=None)
    pmc.add_argument("--progress", action="store_true")

    pg = sub.add_parser("grids", help="9-panel centrality grids (optical & MC if --events>0)")
    pg.add_argument("--system", required=True, choices=["pA","dA","AA"])
    pg.add_argument("--target", required=True, choices=["Au","Pb"])
    pg.add_argument("--energy", type=int, required=True)
    pg.add_argument("--events", type=int, default=0, help="if >0, also make MC grid")
    pg.add_argument("--save", default=None)
    pg.add_argument("--progress", action="store_true")

    args = p.parse_args()
    set_log_level("INFO" if getattr(args, "verbose", False) else "WARNING")

    if args.cmd == "centrality":
        bins = [int(x) for x in args.bins.split(",")]
        opt = OpticalModel(system=args.system, target=args.target, energy_GeV=args.energy,
                           rect_pA=args.rect_pa, verbose=getattr(args, "verbose", False))
        print(f"sigma_tot [{args.units}]:", opt.sigma_tot(units=args.units))
        rows = centrality_table(opt, bins=bins, progress=getattr(args, "progress", False))
        if args.save:
            save_table(rows, args.save); print("Saved:", args.save)
        if args.curves:
            export_optical_curves(opt, out=args.curves); print("Saved curves:", args.curves)
        print(rows[:2] + [rows[-1]])
    elif args.cmd == "mc":
        bins = [int(x) for x in args.bins.split(",")]
        mc = MonteCarloModel(system=args.system, target=args.target, energy_GeV=args.energy)
        res = mc.run(Nevents=args.events, progress=getattr(args, "progress", False))
        print(res.summary())
        rows = centrality_table_mc(res, bins=bins)
        if args.save:
            if args.save.endswith(".json") or args.save.endswith(".csv"):
                save_table(rows, args.save)
            else:
                Path(args.save).write_text(json.dumps({"summary": res.summary(), "table": rows}, indent=2), encoding="utf-8")
            print("Saved:", args.save)
        print(rows[:2] + [rows[-1]])
    elif args.cmd == "grids":
        opt = OpticalModel(system=args.system, target=args.target, energy_GeV=args.energy)
        bins = [(0,5),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,100)]
        fig,_ = centrality_grid_optical(opt, bins=bins)
        if args.events>0:
            mc = MonteCarloModel(system=args.system, target=args.target, energy_GeV=args.energy)
            res = mc.run(Nevents=args.events, progress=getattr(args, "progress", False))
            fig2,_ = centrality_grid_mc(mc, res, bins=bins)
        if args.save:
            import matplotlib.pyplot as plt
            fig.savefig(args.save, dpi=160)
            if args.events>0:
                stem = Path(args.save)
                fig2.savefig(stem.with_name(stem.stem + "_mc" + stem.suffix), dpi=160)
            print("Saved grid(s) to", Path(args.save).resolve())
