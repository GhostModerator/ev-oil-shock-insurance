"""
ISE 599 Project 2 - Step 1: data feasibility check
Pull Chicago TNP (rideshare) vehicle records, classify powertrain from make/model/year,
and compute the monthly share of BEV / hybrid vehicles around the oil price shocks.

Run:  pip install requests pandas matplotlib
      python src/step1_tnp_powertrain_share.py      (run from the repo root)

Outputs (data/processed/ and figures/):
  tnp_vehicle_agg.csv          month x make x model x year counts (raw pull, cached)
  monthly_powertrain_share.csv monthly shares by vehicle count and by trips
  top_ice_models.csv           most common models classified ICE (eyeball for misclassification)
  ambiguous_models.csv         models that could be gas OR hybrid (cannot tell from the record)
  powertrain_share.png         plot with shock windows shaded
"""
import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests

DATASET = "bc6b-sq4u"
RESOURCE = f"https://data.cityofchicago.org/resource/{DATASET}.json"
META = f"https://data.cityofchicago.org/api/views/{DATASET}.json"
APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")  # optional, avoids throttling
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "figures")
for d in (RAW, OUT, FIG):
    os.makedirs(d, exist_ok=True)
CACHE = os.path.join(RAW, "tnp_vehicle_agg.csv")

# Shock windows to shade on the plot (YYYY-MM). Fill in the 2026 window once we pin it down.
SHOCKS = {
    "2022 gas spike": ("2022-03", "2022-07"),
    # War began Feb 28 2026; US avg $2.98 -> $4.15 by Sept 7 and still high. Vehicle data may stop before this.
    "2026 Iran shock": ("2026-03", "2026-10"),
}


# ---------------------------------------------------------------- download
def get(url, params=None):
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=300)
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code}: {r.text[:200]} -- retrying")
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def show_schema():
    meta = get(META)
    print("Dataset:", meta.get("name"))
    print("Columns:")
    for c in meta["columns"]:
        print(f"  {c['fieldName']:<25} {c.get('dataTypeName')}")
    return {c["fieldName"] for c in meta["columns"]}


def pull_aggregated(fields):
    """Aggregate on the server so we never download millions of raw rows."""
    if os.path.exists(CACHE):
        print(f"Using cached {CACHE} (delete it to re-download)")
        return pd.read_csv(CACHE, dtype=str)

    has_trips = "number_of_trips" in fields
    select = "month_reported, make, model, year, count(*) AS n_vehicles"
    if has_trips:
        select += ", sum(number_of_trips) AS n_trips"

    rows, offset, page = [], 0, 50000
    while True:
        params = {
            "$select": select,
            "$group": "month_reported, make, model, year",
            "$order": "month_reported, make, model, year",
            "$limit": page,
            "$offset": offset,
        }
        batch = get(RESOURCE, params)
        rows.extend(batch)
        print(f"  pulled {len(rows):,} grouped rows")
        if len(batch) < page:
            break
        offset += page

    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df.astype(str)


def pull_trip_distribution(fields):
    """How many vehicles had k trips in each month. Used to define driver types."""
    path = os.path.join(RAW, "tnp_trips_per_vehicle_dist.csv")
    if os.path.exists(path):
        print(f"Using cached {path}")
        return
    if "number_of_trips" not in fields:
        print("No number_of_trips column; skipping trips-per-vehicle distribution")
        return
    rows, offset, page = [], 0, 50000
    while True:
        batch = get(RESOURCE, {
            "$select": "month_reported, number_of_trips, count(*) AS n_vehicles",
            "$group": "month_reported, number_of_trips",
            "$order": "month_reported, number_of_trips",
            "$limit": page,
            "$offset": offset,
        })
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path} ({len(rows):,} rows)")


# ---------------------------------------------------------------- classify
def norm(s):
    s = "" if s is None or str(s).lower() == "nan" else str(s)
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


BEV_MAKES = {"TESLA", "POLESTAR", "RIVIAN", "LUCID"}
BEV_MODEL = re.compile(
    r"\b(BOLT|LEAF|MACH E|MACHE|ID 4|ID4|IONIQ 5|IONIQ5|IONIQ 6|IONIQ6|EV6|EV9|"
    r"E TRON|ETRON|I3|I4|IX|BZ4X|SOLTERRA|LYRIQ|ARIYA|E GOLF|EGOLF|"
    r"KONA (EV|ELECTRIC)|NIRO EV|BLAZER EV|EQUINOX EV|MODEL [3SXY])\b"
)
HYBRID_MODEL = re.compile(
    r"\b(HYBRID|HEV|PHEV|PLUG IN|ENERGI|PRIUS|INSIGHT|VOLT|CLARITY|"
    r"(CT|ES|RX|NX|UX|LS|GS|IS|RC) ?\d{3} ?H)\b"
)
# Nameplates sold in BOTH gas and hybrid versions: the record alone cannot tell them apart.
AMBIGUOUS_MODEL = re.compile(
    r"\b(CAMRY|COROLLA|RAV ?4|HIGHLANDER|SIENNA|VENZA|ACCORD|CR ?V|CIVIC|SONATA|"
    r"ELANTRA|TUCSON|SANTA FE|NIRO|IONIQ|KONA|ESCAPE|FUSION|MAVERICK|SPORTAGE|"
    r"SORENTO|PACIFICA|OUTLANDER|MKZ|ES|NX|RX|UX)\b"
)


def classify(make, model, year):
    mk, md = norm(make), norm(model)
    y = pd.to_numeric(year, errors="coerce")
    if mk in BEV_MAKES or BEV_MODEL.search(md) or re.search(r"\b(EV|ELECTRIC)\b", md):
        return "BEV"
    if HYBRID_MODEL.search(md):
        return "HYBRID"
    # hybrid-only model years
    if re.search(r"\bSIENNA\b", md) and y >= 2021:
        return "HYBRID"
    if re.search(r"\bVENZA\b", md) and y >= 2021:
        return "HYBRID"
    if AMBIGUOUS_MODEL.search(md):
        return "AMBIGUOUS"
    return "ICE"


# ---------------------------------------------------------------- analyze
def main():
    fields = show_schema()
    df = pull_aggregated(fields)
    pull_trip_distribution(fields)

    df["month"] = pd.to_datetime(df["month_reported"].str[:7], format="%Y-%m", errors="coerce")
    df["n_vehicles"] = pd.to_numeric(df["n_vehicles"], errors="coerce").fillna(0)
    df["n_trips"] = pd.to_numeric(df.get("n_trips", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["month"])

    # classify each unique make/model/year once
    keys = df[["make", "model", "year"]].drop_duplicates()
    keys["cls"] = [classify(a, b, c) for a, b, c in keys.itertuples(index=False)]
    df = df.merge(keys, on=["make", "model", "year"], how="left")

    print(f"\nMonths covered: {df.month.min():%Y-%m} to {df.month.max():%Y-%m}")
    print(f"Distinct make/model strings: {df[['make','model']].drop_duplicates().shape[0]:,}")
    print("Vehicle-month records by class:")
    print(df.groupby("cls")["n_vehicles"].sum().map("{:,.0f}".format))

    # monthly shares (by vehicles and by trips)
    out = []
    for w in ["n_vehicles", "n_trips"]:
        p = df.pivot_table(index="month", columns="cls", values=w, aggfunc="sum").fillna(0)
        for c in ["BEV", "HYBRID", "AMBIGUOUS", "ICE"]:
            p[c] = p.get(c, 0)
        tot = p[["BEV", "HYBRID", "AMBIGUOUS", "ICE"]].sum(axis=1)
        share = pd.DataFrame({
            f"total_{w}": tot,
            f"bev_share_{w}": p["BEV"] / tot,
            f"hybrid_low_{w}": p["HYBRID"] / tot,                       # confirmed hybrids only
            f"hybrid_high_{w}": (p["HYBRID"] + p["AMBIGUOUS"]) / tot,   # if every ambiguous car were hybrid
        })
        out.append(share)
    monthly = pd.concat(out, axis=1)
    monthly.to_csv(os.path.join(OUT, "monthly_powertrain_share.csv"))

    # sanity outputs
    by_model = df.groupby(["make", "model", "cls"])["n_vehicles"].sum().reset_index()
    by_model[by_model.cls == "ICE"].nlargest(60, "n_vehicles").to_csv(os.path.join(OUT, "top_ice_models.csv"), index=False)
    by_model[by_model.cls == "AMBIGUOUS"].nlargest(60, "n_vehicles").to_csv(os.path.join(OUT, "ambiguous_models.csv"), index=False)

    # suspicious months (quarterly batching, Q2 2018 quirk, partial latest quarter)
    med = monthly["total_n_vehicles"].median()
    odd = monthly[(monthly["total_n_vehicles"] < 0.5 * med) | (monthly["total_n_vehicles"] > 1.5 * med)]
    if len(odd):
        print("\nMonths with unusual vehicle counts (check before trusting):")
        print(odd["total_n_vehicles"].map("{:,.0f}".format))

    print("\nShares around the 2022 shock and the latest months:")
    cols = ["total_n_vehicles", "bev_share_n_vehicles", "hybrid_low_n_vehicles",
            "hybrid_high_n_vehicles", "bev_share_n_trips"]
    view = monthly.loc["2021-07":"2023-06", cols]
    print(view.to_string(float_format=lambda x: f"{x:.4f}" if x < 1 else f"{x:,.0f}"))
    print("\nLast 12 months:")
    print(monthly[cols].tail(12).to_string(float_format=lambda x: f"{x:.4f}" if x < 1 else f"{x:,.0f}"))

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly["bev_share_n_vehicles"], label="BEV (by vehicles)", lw=2)
    ax.plot(monthly.index, monthly["bev_share_n_trips"], label="BEV (by trips)", lw=1, ls="--")
    ax.plot(monthly.index, monthly["hybrid_low_n_vehicles"], label="Hybrid, confirmed", lw=2)
    ax.fill_between(monthly.index, monthly["hybrid_low_n_vehicles"], monthly["hybrid_high_n_vehicles"],
                    alpha=0.15, label="Hybrid range incl. ambiguous models")
    for name, win in SHOCKS.items():
        if win:
            ax.axvspan(pd.Timestamp(win[0]), pd.Timestamp(win[1]) + pd.offsets.MonthEnd(0),
                       color="grey", alpha=0.2)
            ax.text(pd.Timestamp(win[0]), ax.get_ylim()[1] * 0.95, name, fontsize=9)
    ax.set_ylabel("Share of active TNP vehicles")
    ax.set_title("Chicago rideshare fleet: BEV and hybrid share by month")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "powertrain_share.png"), dpi=150)
    print("\nSaved outputs to data/processed/ and figures/")


if __name__ == "__main__":
    main()
