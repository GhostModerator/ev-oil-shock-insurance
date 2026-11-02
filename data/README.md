# Data sources

All raw files go in `data/raw/` (not committed, too big). `src/download_data.py` and `src/step1_tnp_powertrain_share.py` fetch them. Checked Nov 1, 2026.

## Core

| # | Dataset | What we use it for | Link | How we get it |
|---|---|---|---|---|
| 1 | Chicago TNP - Vehicles (`bc6b-sq4u`) | make / model / year per active vehicle per month, trips per vehicle → fleet powertrain mix | https://data.cityofchicago.org/d/bc6b-sq4u | `step1_tnp_powertrain_share.py`, aggregated on the server |
| 2 | Chicago TNP - Trips 2018-2022 (`m6dm-c72p`) | monthly trips, miles, hours, fares, extra charges → demand and revenue per mile | https://data.cityofchicago.org/d/m6dm-c72p | `download_data.py trips hourly` |
| 3 | Chicago TNP - Trips 2023-2024 (`n26f-ihde`) | same | https://data.cityofchicago.org/d/n26f-ihde | same |
| 4 | Chicago TNP - Trips 2025- (`6dvr-xwnh`) | same; updated Oct 27 2026, so it covers most of the 2026 shock | https://data.cityofchicago.org/d/6dvr-xwnh | same |
| 5 | Chicago TNP - Drivers (`j6wf-834c`) | active drivers per month (supply side during shocks) | https://data.cityofchicago.org/d/j6wf-834c | `download_data.py drivers` |
| 6 | EIA weekly retail regular gasoline, Chicago (`EMM_EPMR_PTE_YORD_DPG`) | fit the price models, build scenarios | https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMR_PTE_YORD_DPG&f=W | `download_data.py gas` |
| 7 | EIA weekly retail regular gasoline, Midwest PADD 2 (`EMM_EPMR_PTE_R20_DPG`) and US (FRED `GASREGW`) | backups if the Chicago series has gaps | https://fred.stlouisfed.org/series/GASREGW | `download_data.py gas` |
| 8 | Brent crude spot, daily (FRED `DCOILBRENTEU`, source EIA) | sharper view of jump timing and size | https://fred.stlouisfed.org/series/DCOILBRENTEU | `download_data.py brent` |
| 9 | EIA monthly retail electricity price, Illinois (residential, commercial) | EV charging cost, and its correlation with gas | https://www.eia.gov/electricity/data/browser/ | `download_data.py power` (free EIA API key) |
| 10 | fueleconomy.gov vehicle table | MPG / kWh per 100 mi by make-model-year, and the reference list for powertrain labels | https://www.fueleconomy.gov/feg/download.shtml | `download_data.py mpg` |

## Optional

| Dataset | Use | Link |
|---|---|---|
| AFDC charging stations (NREL) | public fast-charger count in Chicago, a charging constraint | https://afdc.energy.gov/stations |
| Vehicle purchase prices and resale values | first-stage capital cost, end-of-horizon resale | published cost-of-ownership studies + a few listings, entered by hand |

## Known issues

- TNP data is reported quarterly, so the newest quarter may be missing. The vehicles table may cover only the start of the 2026 shock (war began Feb 28 2026). The trips table runs into fall 2026.
- Q2 2018 vehicle records were filed as a quarter and put in June 2018.
- Vehicle records often list the model only (e.g. "CAMRY"), so gas vs hybrid versions of the same nameplate can't be told apart. Step 1 reports hybrid share as a range for this reason.
- No persistent vehicle ID, so we can't follow one car over time. Recent model years stand in for new entries.
- Fares are rounded to $2.50 and trip times to 15 minutes for privacy.
