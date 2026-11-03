"""Figures for the paper. Run from repo root: python paper/make_figures.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
BLUE, GREY, RED = "#1f4e79", "#9a9a9a", "#b03a2e"

# 1. fleet share
s = pd.read_csv(os.path.join(ROOT, "data/labels/monthly_share_step2.csv"), index_col=0, parse_dates=True)["2019-01":]
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.fill_between(s.index, s.hybrid_low * 100, s.hybrid_high * 100, color=GREY, alpha=0.35, lw=0, label="Hybrid (range)")
ax.plot(s.index, s.bev * 100, color=BLUE, lw=1.6, label="Battery electric")
ax.plot(s.index, s.phev * 100, color=BLUE, lw=1, ls="--", label="Plug-in hybrid")
for d, lab, ha in (("2022-03-01", " Ukraine", "left"), ("2026-03-01", "Iran war ", "right")):
    ax.axvline(pd.Timestamp(d), color=RED, lw=0.8, ls=":")
    ax.text(pd.Timestamp(d), 49, lab, color=RED, fontsize=8, va="top", ha=ha)
ax.set_ylabel("Share of active vehicles (%)")
ax.set_ylim(0, 50)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
fig.savefig(os.path.join(OUT, "fleet_share.pdf"))

# 2. revenue per hour by clock hour
tb = pd.read_csv(os.path.join(ROOT, "data/params/time_blocks.csv")).sort_values("hour")
fig, ax = plt.subplots(figsize=(6, 2.4))
ax.bar(tb.hour, tb.fare_rev_per_hour, color=GREY, label="Fare (driver share)")
ax.bar(tb.hour, tb.tip_per_hour, bottom=tb.fare_rev_per_hour, color=BLUE, label="Tips")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (trip start)")
ax.set_ylabel("Driver revenue per\nlogged-in hour (\\$)".replace("\\$", "$"))
ax.set_ylim(0, 23)
ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center")
fig.savefig(os.path.join(OUT, "revenue_by_hour.pdf"))

# 3. gas backtest
sys.path.insert(0, os.path.join(ROOT, "src"))
import step4_gas_models as g4
g = g4.load_weekly()
fit = g[g4.FIT_START:"2025-12-31"]
p, seas, _ = g4.fit_jump(fit)
pb, seas_m = g4.fit_regime(fit)
n = 5000
m_act = g["2026-01-01":].resample("MS").mean()
simA = g4.simulate_jump(p, seas, fit.iloc[-1], fit.index[-1], 48 * 52 // 12, n,
                        prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=1).resample("MS").mean()
simB, _ = g4.simulate_regime(pb, seas_m, fit.resample("MS").mean(), 48, n, seed=2)
hist = fit["2022-01-01":].resample("MS").mean()
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
for ax, sim, title in ((axes[0], simA, "(a) Model A: mean reversion with jumps"),
                       (axes[1], simB, "(b) Model B: calm/crisis regimes")):
    q = sim.loc[:"2027-12"].quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    ax.fill_between(q.index, q[0.05], q[0.95], color=BLUE, alpha=0.15, lw=0, label="90% band")
    ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE, alpha=0.3, lw=0, label="50% band")
    ax.plot(q.index, q[0.5], color=BLUE, lw=0.9, label="Median")
    ax.plot(hist.index, hist.values, color="black", lw=1, label="Observed (fit)")
    ax.plot(m_act.index, m_act.values, color=RED, lw=1.6, label="Observed 2026 (held out)")
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel("Chicago regular gasoline\n($/gal, monthly mean)")
import matplotlib.dates as mdates
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
fig.savefig(os.path.join(OUT, "gas_backtest.pdf"))

# 4. hybrid gain sweep
w = pd.read_csv(os.path.join(ROOT, "results/step5_sweep_A_spike.csv"))
w = w[w.lam == 0].pivot(index="hours", columns="car", values=["mean_loss", "cvar_loss"])
mean = (w["mean_loss"]["keep_gas"] - w["mean_loss"]["hybrid"]) / 1000
tail = (w["cvar_loss"]["keep_gas"] - w["cvar_loss"]["hybrid"]) / 1000
def cross(v):
    h, y = v.index.values, v.values
    i = np.where((y[:-1] < 0) & (y[1:] >= 0))[0][0]
    return h[i] + (h[i + 1] - h[i]) * (-y[i]) / (y[i + 1] - y[i])
cm, ct = cross(mean), cross(tail)
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.axhline(0, color="black", lw=0.6)
ax.axvspan(cm, ct, color=GREY, alpha=0.3, lw=0)
ax.plot(mean.index, mean.values, "o-", color=GREY, ms=3, lw=1.4, label="Expected gain")
ax.plot(tail.index, tail.values, "s-", color=BLUE, ms=3, lw=1.4, label="Gain in worst 10% (CVaR$_{0.9}$)")
ax.annotate(f"{cm:.0f}-{ct:.0f} h/month:\nchoice depends on $\\lambda$", xy=((cm + ct) / 2, -2), xytext=(110, -5.5),
            fontsize=8, arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
ax.set_xlabel("Hours driven per month (baseline)")
ax.set_ylabel("4-year gain of hybrid over\nkeeping the gas car ($000)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(os.path.join(OUT, "hybrid_sweep.pdf"))
print("crossings", cm, ct)
