"""
Step 3b: driver types and the hour-of-day revenue curve.

Driver types come from the TNP Vehicles data: how many trips each car gave in a month.
We use a car-month as a stand-in for a driver-month.

The revenue curve comes from the TNP Trips data, aggregated by month and hour of day.
For each hour h we compute, per hour the driver is logged in:
    fare to driver  = fare per trip-hour  x  utilization_h  x  driver share of fare
    miles driven    = trip mph            x  utilization_h  /  paid-mile share
Utilization (share of logged-in time with a rider) averages U_BAR. Busy hours keep drivers
busier, so utilization_h = U_BAR * (demand_h / mean demand) ** BETA, scaled back to average U_BAR.

Run from repo root:  python src/step3_drivers.py
Outputs: data/params/driver_types.csv, data/params/time_blocks.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FETCHED = os.path.join(ROOT, "data", "fetched")
PARAMS = os.path.join(ROOT, "data", "params")

U_BAR = 0.50     # Cramer & Krueger (2016): UberX drivers have a rider about half of logged-in time
BETA = 0.3       # how much busier drivers are in busy hours (0 = flat; sensitivity 0 to 0.6)
HOUR_CAP = 260   # one driver can't log much more than 60 hours a week; above that we assume a shared car
FLEX = 1.2       # a driver can stretch to 20% above usual hours (up to HOUR_CAP)
BINS = [(1, 50, "light"), (51, 200, "regular"), (201, 400, "heavy"), (401, 10**6, "full_time")]
DAYS = 365.25 / 12


def market():
    m = pd.read_csv(os.path.join(PARAMS, "market.csv")).set_index("param")["base"]
    return float(m["driver_share_of_fare"]), float(m["paid_mile_share"])


def load_trips(n_months=12):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FETCHED, "trips", "tnp_trips_month_hour_*.csv")))])
    full = d.groupby("month")["hour"].nunique().loc[lambda s: s == 24].index   # months with all 24 hours
    months = sorted(full)[-n_months:]
    return d[d.month.isin(months)], months


def tip_ratio(months):
    """Tips / fare by hour of day, from the tips files. Tips go to the driver in full."""
    files = sorted(glob.glob(os.path.join(FETCHED, "trips", "tnp_tips_month_hour_*.csv")))
    if not files:
        print("No tips files yet: tips set to 0")
        return pd.Series(0.0, index=range(24))
    t = pd.concat([pd.read_csv(f) for f in files])
    t = t[t.month.isin(months)] if t.month.isin(months).any() else t
    g = t.groupby("hour")[["tip", "fare"]].sum()
    print(f"Tips: {sorted(t.month.unique())[0][:7]} to {sorted(t.month.unique())[-1][:7]}, "
          f"{g.tip.sum() / g.fare.sum():.1%} of fare")
    return g.tip / g.fare


def time_blocks(d, share, paid, tips):
    h = d.groupby("hour")[["trips", "miles", "seconds", "fare"]].sum()
    trip_hr = h.seconds / 3600
    demand = h.trips / h.trips.mean()
    u = demand ** BETA
    u = U_BAR * u / np.average(u, weights=trip_hr)
    u = u.clip(0.2, 0.8)
    out = pd.DataFrame({
        "hour": h.index,
        "trip_share": h.trips / h.trips.sum(),
        "fare_per_trip": h.fare / h.trips,
        "fare_per_trip_hour": h.fare / trip_hr,
        "utilization": u,
        "fare_rev_per_hour": h.fare / trip_hr * u * share,
        "tip_per_hour": h.fare / trip_hr * u * tips.reindex(h.index).fillna(0),
        "miles_per_hour": h.miles / trip_hr * u / paid,
        "trips_per_hour": h.trips / trip_hr * u,
        "hours_available": DAYS,          # each clock hour comes round about 30.4 times a month
    }).reset_index(drop=True)
    out["driver_rev_per_hour"] = out.fare_rev_per_hour + out.tip_per_hour
    out = out.sort_values("driver_rev_per_hour", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    out["cum_hours"] = out["hours_available"].cumsum()
    return out


def driver_types(tb):
    v = pd.read_csv(os.path.join(FETCHED, "tnp_trips_per_vehicle_dist.csv"))
    v = v[v.number_of_trips > 0]
    last = sorted(v.month_reported.unique())[-12:]      # latest 12 months of vehicle reports
    v = v[v.month_reported.isin(last)]
    tot_cars, tot_trips = v.n_vehicles.sum(), (v.n_vehicles * v.number_of_trips).sum()
    trips_per_hr = np.average(tb.trips_per_hour, weights=tb.trip_share / tb.trips_per_hour)  # time-weighted
    rows = []
    for lo, hi, name in BINS:
        g = v[v.number_of_trips.between(lo, hi)]
        n = g.n_vehicles.sum()
        trips = (g.n_vehicles * g.number_of_trips).sum() / n
        hours = trips / trips_per_hr
        rows.append({
            "type": name, "trips_lo": lo, "trips_hi": hi if hi < 10**6 else None,
            "share_of_cars": n / tot_cars, "share_of_trips": (g.n_vehicles * g.number_of_trips).sum() / tot_trips,
            "cars_per_month": n / len(last), "trips_per_month": trips,
            "hours_per_month": hours, "hours_per_week": hours / (DAYS / 7),
        })
    t = pd.DataFrame(rows)
    # miles: each type drives its best hours first (the curve), so use the curve to get miles
    miles = []
    for H in t.hours_per_month:
        left, m = H, 0.0
        for r in tb.itertuples():
            take = min(left, r.hours_available)
            m += take * r.miles_per_hour
            left -= take
            if left <= 0:
                break
        miles.append(m)
    t["miles_per_month"] = miles
    t["vehicle_months_used"] = f"{last[0]} to {last[-1]}"
    return t


def gas_cost_per_mile(months):
    c = pd.read_csv(os.path.join(PARAMS, "cars.csv")).set_index("param")["keep_gas"]
    g = pd.read_csv(os.path.join(FETCHED, "eia_gas_chicago_weekly.csv"))
    g.columns = ["date", "price"]
    g = g[g.date.between(months[0][:7], months[-1][:7] + "-31")]
    p = g.price.mean()
    return p, p / (float(c["mpg_epa"]) * float(c["city_mpg_factor"])) + float(c["maintenance"])


def calibrate(t, tb, cpm):
    """Reservation wage per type: the net value of the last hour each type drives today (gas car).

    If drivers pick their best hours and stop when an hour is worth less than their outside option,
    then the hours they drive now tell us that outside option.
    """
    b = tb.assign(net=tb.driver_rev_per_hour - tb.miles_per_hour * cpm).sort_values("net", ascending=False)
    rows = []
    for r in t.itertuples():
        H = min(r.hours_per_month, HOUR_CAP)
        left, miles, rev = H, 0.0, 0.0
        for x in b.itertuples():
            take = min(left, x.hours_available)
            miles += take * x.miles_per_hour
            rev += take * x.driver_rev_per_hour
            left -= take
            if left <= 1e-9:
                break
        rows.append({"hours_base": H, "hours_max": min(FLEX * H, HOUR_CAP), "cars_per_driver_month": r.hours_per_month / H,
                     "miles_base": miles, "revenue_base": rev, "reservation_wage": x.net})
    return t.drop(columns=["miles_per_month"]).join(pd.DataFrame(rows, index=t.index))


def main():
    share, paid = market()
    d, months = load_trips()
    print(f"Trip months used: {months[0]} to {months[-1]} ({len(months)})")
    tb = time_blocks(d, share, paid, tip_ratio(months))
    min_per_trip = d.seconds.sum() / d.trips.sum() / 60
    t = driver_types(tb)
    gas, cpm = gas_cost_per_mile(months)
    print(f"Chicago gas over these months ${gas:.2f}/gal -> gas-car cost ${cpm:.3f}/mi (fuel + maintenance)")
    t = calibrate(t, tb, cpm)
    tb.to_csv(os.path.join(PARAMS, "time_blocks.csv"), index=False)
    t.to_csv(os.path.join(PARAMS, "driver_types.csv"), index=False)
    print(f"Minutes per trip {min_per_trip:.1f}, miles per trip {d.miles.sum() / d.trips.sum():.2f}, "
          f"fare per trip ${d.fare.sum() / d.trips.sum():.2f}")
    print("\nHour-of-day blocks, best first:")
    print(tb[["rank", "hour", "trip_share", "fare_per_trip_hour", "utilization", "tip_per_hour", "driver_rev_per_hour",
              "miles_per_hour", "cum_hours"]].round(3).to_string(index=False))
    print("\nDriver types:")
    print(t.drop(columns=["vehicle_months_used"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
