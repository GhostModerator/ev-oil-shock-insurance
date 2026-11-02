"""
Download every dataset the project needs into data/raw/.

Run from the repo root:
    pip install -r requirements.txt
    python src/download_data.py            # everything
    python src/download_data.py gas power  # only some parts

Parts:
    licenses Chicago public passenger vehicle licenses (taxis); may list fuel type
    trips    Chicago TNP trips by month x hour of day (count, miles, seconds, fare, extra charges),
             one file per year, plus monthly totals. "trips:2022" runs one year.
    hourly   same month x hour files, without rebuilding the monthly totals
    drivers  Chicago TNP drivers, active drivers per month
    gas      EIA weekly retail regular gasoline: Chicago, Midwest (PADD 2), US average
    brent    Brent crude spot price, daily (FRED DCOILBRENTEU, source EIA)
    power    EIA monthly retail electricity price, Illinois (residential + commercial)
    mpg      fueleconomy.gov vehicle table (MPG / kWh per 100 mi by make-model-year)

The TNP vehicles table is pulled by src/step1_tnp_powertrain_share.py.

Optional environment variables:
    SOCRATA_APP_TOKEN   Chicago data portal token (avoids throttling)
    EIA_API_KEY         free key from https://www.eia.gov/opendata/register.php (needed for "power")
"""
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

SOCRATA = "https://data.cityofchicago.org/resource/{}.json"
TOKEN = os.environ.get("SOCRATA_APP_TOKEN")
EIA_KEY = os.environ.get("EIA_API_KEY")

TRIP_DATASETS = {           # dataset id -> years it covers
    "m6dm-c72p": range(2018, 2023),   # Trips (2018-2022)
    "n26f-ihde": range(2023, 2025),   # Trips (2023-2024)
    "6dvr-xwnh": range(2025, 2027),   # Trips (2025-)
}
DRIVERS = "j6wf-834c"


def socrata(dataset, params):
    headers = {"X-App-Token": TOKEN} if TOKEN else {}
    for attempt in range(6):
        try:
            r = requests.get(SOCRATA.format(dataset), params=params, headers=headers, timeout=600)
        except requests.RequestException as e:          # dropped connection: the server is still busy, retry
            print(f"    {type(e).__name__} -- retry {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 400:                        # bad query (e.g. missing column): don't retry
            r.raise_for_status()
        print(f"    HTTP {r.status_code}: {r.text[:150]} -- retry {attempt + 1}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError("Socrata request kept failing")


BASE_SELECT = ("count(*) AS trips, sum(trip_miles) AS miles, "
               "sum(trip_seconds) AS seconds, sum(fare) AS fare")


def year_where(y):
    return (f"trip_start_timestamp >= '{y}-01-01T00:00:00' "
            f"AND trip_start_timestamp < '{y + 1}-01-01T00:00:00'")


def months_of(y):
    last = pd.Timestamp.today().to_period("M") - 1          # skip the current, incomplete month
    return [m for m in pd.period_range(f"{y}-01", f"{y}-12", freq="M") if m <= last]


def dataset_for(y):
    return next(ds for ds, years in TRIP_DATASETS.items() if y in years)


def get_trips_hourly(years=None):
    """Month x hour-of-day totals, one query per month (a year at once is too slow).

    Each year goes to its own file and is saved after every month, so a run that stops
    part-way keeps what it has and a re-run only fetches the missing months.
    Monthly totals = sum over hours, so no separate monthly query is needed.
    """
    years = years or [y for yrs in TRIP_DATASETS.values() for y in yrs]
    for y in years:
        ds = dataset_for(y)
        f = os.path.join(RAW, f"tnp_trips_month_hour_{y}.csv")
        have = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
        if len(have) and "additional_charges" in have:     # re-fetch months saved without the fee column
            have = have[have.groupby("month")["additional_charges"].transform(lambda c: c.notna().any())]
        done = set(have["month"].str[:7]) if len(have) else set()
        for m in months_of(y):
            if str(m) in done:
                continue
            nxt = m + 1
            t0 = time.time()
            where = (f"trip_start_timestamp >= '{m}-01T00:00:00' "
                     f"AND trip_start_timestamp < '{nxt}-01T00:00:00'")
            batch = []
            for extra in (", sum(additional_charges) AS additional_charges", ""):
                try:
                    batch = socrata(ds, {
                        "$select": f"date_extract_hh(trip_start_timestamp) AS hour, {BASE_SELECT}{extra}",
                        "$where": where, "$group": "hour", "$order": "hour", "$limit": 100})
                    break
                except Exception as e:
                    print(f"    {m} failed{' with additional_charges' if extra else ''}: {e}", flush=True)
            if not batch:
                continue
            b = pd.DataFrame(batch)
            b.insert(0, "month", f"{m}-01")
            b["dataset"] = ds
            have = pd.concat([have, b], ignore_index=True)
            have.to_csv(f, index=False)
            print(f"  {ds} {m}: {int(b['trips'].astype(float).sum()):,} trips ({time.time() - t0:.0f}s)", flush=True)


def get_tips(years=None):
    """Tips by month x hour of day (the hourly files have fare but not tips). One query per month."""
    years = years or [2024, 2025, 2026]
    for y in years:
        ds = dataset_for(y)
        cols = list(socrata(ds, {"$limit": 1})[0].keys())
        print(f"  {ds} columns: {cols}", flush=True)
        tipcol = next((c for c in cols if c.lower().startswith("tip")), None)
        if tipcol is None:
            print(f"  {ds} has no tip column; skipping {y}", flush=True)
            continue
        f = os.path.join(RAW, f"tnp_tips_month_hour_{y}.csv")
        have = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
        done = set(have["month"].str[:7]) if len(have) else set()
        for m in months_of(y):
            if str(m) in done:
                continue
            t0 = time.time()
            where = (f"trip_start_timestamp >= '{m}-01T00:00:00' "
                     f"AND trip_start_timestamp < '{m + 1}-01T00:00:00'")
            try:
                batch = socrata(ds, {
                    "$select": "date_extract_hh(trip_start_timestamp) AS hour, count(*) AS trips, "
                               f"sum(fare) AS fare_sum, sum({tipcol}) AS tip_sum",
                    "$where": where, "$group": "hour", "$order": "hour", "$limit": 100})
            except Exception as e:
                print(f"    {m} failed: {e}", flush=True)
                continue
            b = pd.DataFrame(batch).rename(columns={"fare_sum": "fare", "tip_sum": "tip"})
            b.insert(0, "month", f"{m}-01")
            b["dataset"] = ds
            have = pd.concat([have, b], ignore_index=True)
            have.to_csv(f, index=False)
            print(f"  {ds} {m}: tips saved ({time.time() - t0:.0f}s)", flush=True)


def get_trips(years=None):
    """Monthly totals, summed from the month x hour files."""
    get_trips_hourly(years)
    files = [os.path.join(RAW, x) for x in sorted(os.listdir(RAW)) if x.startswith("tnp_trips_month_hour_")]
    if not files:
        return
    h = pd.concat([pd.read_csv(x) for x in files])
    num = [c for c in ("trips", "miles", "seconds", "fare", "additional_charges") if c in h]
    h[num] = h[num].apply(pd.to_numeric, errors="coerce")
    df = h.groupby(["month", "dataset"], as_index=False)[num].sum()
    df.to_csv(os.path.join(RAW, "tnp_trips_monthly.csv"), index=False)
    print(f"  saved tnp_trips_monthly.csv ({len(df)} months)")


def get_drivers():
    batch = socrata(DRIVERS, {
        "$select": "month_reported, count(*) AS drivers",
        "$group": "month_reported",
        "$order": "month_reported",
        "$limit": 5000,
    })
    df = pd.DataFrame(batch)
    df.to_csv(os.path.join(RAW, "tnp_drivers_monthly.csv"), index=False)
    print(f"  saved tnp_drivers_monthly.csv ({len(df)} months)")


def eia_xls(series, col, fname, freq="w"):
    """EIA history spreadsheet for one series (freq w = weekly, d = daily). Returns True if saved."""
    url = f"https://www.eia.gov/dnav/pet/hist_xls/{series}{freq}.xls"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
        df.columns = ["date", col]
        df.to_csv(os.path.join(RAW, fname), index=False)
        print(f"  saved {fname} ({len(df)} rows, {df['date'].min()} to {df['date'].max()})", flush=True)
        return True
    except Exception as e:
        print(f"  EIA {series} failed ({e})", flush=True)
        return False


def fred(series, col, fname):
    """FRED csv. Returns True if saved."""
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", col]
        df.to_csv(os.path.join(RAW, fname), index=False)
        print(f"  saved {fname} ({len(df)} rows, {df['date'].min()} to {df['date'].max()})", flush=True)
        return True
    except Exception as e:
        print(f"  FRED {series} failed ({e})", flush=True)
        return False


def get_gas():
    # Chicago is the main series; Midwest (PADD 2) and US are backups / comparisons
    eia_xls("EMM_EPMR_PTE_YORD_DPG", "chicago_regular_usd_gal", "eia_gas_chicago_weekly.csv")
    eia_xls("EMM_EPMR_PTE_R20_DPG", "midwest_regular_usd_gal", "eia_gas_midwest_weekly.csv")
    eia_xls("EMM_EPMR_PTE_NUS_DPG", "us_regular_usd_gal", "eia_gas_us_weekly.csv") or \
        fred("GASREGW", "us_regular_usd_gal", "eia_gas_us_weekly.csv")


def get_brent():
    eia_xls("RBRTE", "brent_usd_bbl", "brent_daily.csv", freq="d") or \
        fred("DCOILBRENTEU", "brent_usd_bbl", "brent_daily.csv")


def get_power():
    if not EIA_KEY:
        print("  skipped: set EIA_API_KEY first (free, https://www.eia.gov/opendata/register.php)")
        return
    params = {
        "api_key": EIA_KEY,
        "frequency": "monthly",
        "data[0]": "price",
        "facets[stateid][]": "IL",
        "facets[sectorid][]": ["RES", "COM"],
        "start": "2001-01",   # long history so we can see how power moved in 2005, 2008, 2022

        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get("https://api.eia.gov/v2/electricity/retail-sales/data/", params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])
    df.to_csv(os.path.join(RAW, "eia_power_illinois_monthly.csv"), index=False)
    print(f"  saved eia_power_illinois_monthly.csv ({len(df)} rows, price in cents/kWh)")


def get_mpg():
    r = requests.get("https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip", timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(RAW)
    print("  saved vehicles.csv (fueleconomy.gov)")
    # Slim copy with only the columns we use (small enough to commit)
    full = pd.read_csv(os.path.join(RAW, "vehicles.csv"), low_memory=False)
    keep = [c for c in ["year", "make", "model", "baseModel", "atvType", "fuelType", "fuelType1",
                        "comb08", "combE", "evMotor", "VClass", "trany", "id"] if c in full.columns]
    full[full["year"] >= 2005][keep].to_csv(os.path.join(RAW, "epa_vehicles_slim.csv"), index=False)
    print(f"  saved epa_vehicles_slim.csv ({len(keep)} columns, model years 2005+)")


def get_licenses():
    """Chicago public passenger vehicle licenses (taxis etc.). Small; may have a fuel type column."""
    r = requests.get("https://data.cityofchicago.org/api/views/tfm3-3j95/rows.csv?accessType=DOWNLOAD",
                     timeout=300)
    r.raise_for_status()
    path = os.path.join(RAW, "public_passenger_vehicle_licenses.csv")
    with open(path, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(path, nrows=5)
    print(f"  saved public_passenger_vehicle_licenses.csv; columns: {list(df.columns)}")


PARTS = {"licenses": get_licenses, "trips": get_trips, "hourly": get_trips_hourly, "tips": get_tips, "drivers": get_drivers,
         "gas": get_gas, "brent": get_brent, "power": get_power, "mpg": get_mpg}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(PARTS)
    for arg in wanted:                      # "trips:2022" or "trips:2022,2023" limits the years
        name, _, yrs = arg.partition(":")
        print(f"[{arg}]", flush=True)
        try:
            if yrs:
                PARTS[name]([int(y) for y in yrs.split(",")])
            else:
                PARTS[name]()
        except Exception as e:
            print(f"  FAILED: {e}")
