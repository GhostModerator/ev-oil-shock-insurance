"""
Step 5b: where does the best car flip? Sweep driving intensity (hours per month) and lambda.

Each point is a synthetic driver who drives H hours a month today, with the reservation wage
implied by H (same calibration as step 3). For each H we compare the three cars on mean loss
and on CVaR, and find the H where each comparison flips.

Run from repo root:  python src/step5_sweep.py [A_spike|B_regime]
Outputs: results/step5_sweep_{model}.csv, figures/step5_sweep_{model}.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_drivers as s3  # noqa: E402
import step5_model as m  # noqa: E402

GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 190, 220, 260]


def synthetic_types(cpm=None):
    tb = pd.read_csv(os.path.join(s3.PARAMS, "time_blocks.csv"))
    P = m.load_params()
    cpm = cpm or 3.80 / P["keep_gas"]["mpg"] + P["keep_gas"]["maint"]
    t = pd.DataFrame({"hours_per_month": GRID, "miles_per_month": 0.0}, index=[f"H{h}" for h in GRID])
    return s3.calibrate(t, tb, cpm)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "A_spike"
    t = synthetic_types()
    res, _ = m.run(model, types_df=t, lambdas=[0.0, 0.5, 1.0], verbose=False, tag="sweep")
    res["hours"] = res["type"].str[1:].astype(int)
    res.to_csv(os.path.join(m.RES, f"step5_sweep_{model}.csv"), index=False)
    w = res[res.lam == 0].pivot(index="hours", columns="car", values=["mean_loss", "cvar_loss"])
    tab = pd.DataFrame({
        "r": t.set_index(t.hours_per_month.astype(int)).reservation_wage,
        "miles": t.set_index(t.hours_per_month.astype(int)).miles_base,
        "E: keep-hyb": w["mean_loss"]["keep_gas"] - w["mean_loss"]["hybrid"],
        "CVaR: keep-hyb": w["cvar_loss"]["keep_gas"] - w["cvar_loss"]["hybrid"],
        "E: hyb-ev": w["mean_loss"]["hybrid"] - w["mean_loss"]["ev"],
        "CVaR: hyb-ev": w["cvar_loss"]["hybrid"] - w["cvar_loss"]["ev"],
    })
    best = res[res.chosen].pivot(index="hours", columns="lam", values="car")
    tab = tab.join(best.add_prefix("best@lam="))
    print(f"Gas model {model}. Positive = the second car has the lower loss (better).")
    print(tab.round(1).to_string())
    for col in ("E: keep-hyb", "CVaR: keep-hyb"):
        v = tab[col]
        cross = [h for h0, h, a, b in zip(v.index[:-1], v.index[1:], v.values[:-1], v.values[1:]) if a < 0 <= b]
        if cross:
            i = list(v.index).index(cross[0])
            h0, h1, a, b = v.index[i - 1], v.index[i], v.values[i - 1], v.values[i]
            print(f"{col}: hybrid pays off above ~{h0 + (h1 - h0) * (-a) / (b - a):.0f} h/month")
    plot(tab, model)


def plot(tab, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(tab.index, tab["E: keep-hyb"] / 1000, "o-", color="#1f3f66", label="mean gain from hybrid")
    ax.plot(tab.index, tab["CVaR: keep-hyb"] / 1000, "s--", color="#c0392b", label="worst-10% gain from hybrid")
    ax.set_xlabel("hours driven per month today")
    ax.set_ylabel("4-year gain vs keeping the gas car ($000)")
    ax.set_title(f"Hybrid vs keeping the gas car (gas model {model.split('_')[0]})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(m.RES), "figures", f"step5_sweep_{model}.png"), dpi=150)


if __name__ == "__main__":
    main()
