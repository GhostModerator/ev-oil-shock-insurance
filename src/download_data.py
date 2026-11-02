"""
Download every dataset the project needs into data/raw/.

Run from the repo root:
    pip install -r requirements.txt
    python src/download_data.py            # everything
    python src/download_data.py gas power  # only some parts

Parts:
    licenses Chicago public passenger vehicle licenses (taxis); may list fuel type
    trips   Chicago TNP trips, aggregated to monthly totals (count, miles, seconds, fare, extra charges)
    hourly   Chicago TNP trips by month x hour of day (for peak demand vs. charging downtime)
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
    for attempt in range(5):
        r = requests.get(SOCRATA.format(dataset), params=params, headers=headers, timeout=600)
        if r.status_code == 200:
            return r.json()
        print(f"    HTTP {r.status_code}: {r.text[:150]} -- retry {attempt + 1}")
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()


BASE_SELECT = ("count(*) AS trips, sum(trip_miles) AS miles, "
               "sum(trip_seconds) AS seconds, sum(fare) AS fare")


def year_where(y):
    return (f"trip_start_timestamp >= '{y}-01-01T00:00:00' "
            f"AND trip_start_timestamp < '{y + 1}-01-01T00:00:00'")


def trips_query(ds, y, group_cols, group_by):
    """Try with additional_charges (fees/surcharges); fall back if a dataset lacks it."""
    for extra in (", sum(additional_charges) AS additional_charges", ""):
        try:
            return socrata(ds, {
                "$select": f"{group_cols}, {BASE_SELECT}{extra}",
                "$where": year_where(y),
                "$group": group_by,
                "$order": group_by,
                "$limit": 50000,
            })
        except Exception as e:
            print(f"    query failed{' with additional_charges' if extra else ''}: {e}")
    return []


def get_trips():
    """Monthly totals only. The raw trip tables are hundreds of millions of rows."""
    rows = []
    for ds, years in TRIP_DATASETS.items():
        for y in years:
            print(f"  trips {ds} {y}")
            batch = trips_query(ds, y, "date_trunc_ym(trip_start_timestamp) AS month", "month")
            for b in batch:
                b["dataset"] = ds
            rows.extend(batch)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW, "tnp_trips_monthly.csv"), index=False)
    print(f"  saved tnp_trips_monthly.csv ({len(df)} months)")


def get_trips_hourly():
    """Month x hour-of-day totals, to see how much demand falls in peak hours."""
    rows = []
    for ds, years in TRIP_DATASETS.items():
        for y in years:
            print(f"  hourly trips {ds} {y}")
            batch = trips_query(
                ds, y,
                "date_trunc_ym(trip_start_timestamp) AS month, date_extract_hh(trip_start_timestamp) AS hour",
                "month, hour")
            for b in batch:
                b["dataset"] = ds
            rows.extend(batch)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW, "tnp_trips_month_hour.csv"), index=False)
    print(f"  saved tnp_trips_month_hour.csv ({len(df)} rows)")


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


def eia_weekly_xls(series, col, fname):
    """EIA history spreadsheet for one weekly series."""
    url = f"https://www.eia.gov/dnav/pet/hist_xls/{series}w.xls"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
        df.columns = ["date", col]
        df.to_csv(os.path.join(RAW, fname), index=False)
        print(f"  saved {fname} ({len(df)} weeks, {df['date'].min()} to {df['date'].max()})")
    except Exception as e:
        print(f"  {series} download failed ({e}). Download by hand from\n"
              f"  https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s={series}&f=W")


def fred(series, col, fname):
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    df.columns = ["date", col]
    df.to_csv(os.path.join(RAW, fname), index=False)
    print(f"  saved {fname} ({len(df)} rows, {df['date'].min()} to {df['date'].max()})")


def get_gas():
    # Chicago is the main series; Midwest (PADD 2) and US are backups / comparisons
    eia_weekly_xls("EMM_EPMR_PTE_YORD_DPG", "chicago_regular_usd_gal", "eia_gas_chicago_weekly.csv")
    eia_weekly_xls("EMM_EPMR_PTE_R20_DPG", "midwest_regular_usd_gal", "eia_gas_midwest_weekly.csv")
    fred("GASREGW", "us_regular_usd_gal", "eia_gas_us_weekly.csv")


def get_brent():
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


PARTS = {"licenses": get_licenses, "trips": get_trips, "hourly": get_trips_hourly, "drivers": get_drivers,
         "gas": get_gas, "brent": get_brent, "power": get_power, "mpg": get_mpg}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(PARTS)
    for name in wanted:
        print(f"[{name}]")
        try:
            PARTS[name]()
        except Exception as e:
            print(f"  FAILED: {e}")
