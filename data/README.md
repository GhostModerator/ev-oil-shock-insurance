# Data sources

All raw files go in `data/raw/` (not committed, too big). `src/download_data.py` and `src/step1_tnp_powertrain_share.py` fetch them.

## Core

| # | Dataset | What we use it for | Link | How we get it |
|---|---|---|---|---|
| 1 | Chicago TNP – Vehicles (`bc6b-sq4u`) | make / model / year per active vehicle per month, trips per vehicle → fleet powertrain mix | https://data.cityofchicago.org/d/bc6b-sq4u | step 1 script, server-side aggregated |
| 2 | Chicago TNP – Trips 2018–2022 (`m6dm-c72p`) | monthly trips, miles, hours, fares → demand and revenue per mile | https://data.cityofchicago.org/d/m6dm-c72p | `download_data.py trips` (monthly totals only) |
| 3 | Chicago TNP – Trips 2023–2024 (`n26f-ihde`) | same | https://data.cityofchicago.org/d/n26f-ihde | same |
| 4 | Chicago TNP – Trips 2025– (`6dvr-xwnh`) | same | https://data.cityofchicago.org/d/6dvr-xwnh | same |
| 5 | EIA weekly retail regular gasoline, Chicago (`EMM_EPMR_PTE_YORD_DPG`) | fit the jump-diffusion price model, build scenarios | https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMR_PTE_YORD_DPG&f=W | `download_data.py gas` |
| 6 | EIA weekly US regular gasoline (FRED `GASREGW`) | backup series, longer history check | https://fred.stlouisfed.org/series/GASREGW | `download_data.py gas` |
| 7 | EIA monthly retail electricity price, Illinois (residential, commercial) | EV charging cost | https://www.eia.gov/electricity/data/browser/ | `download_data.py power` (needs free EIA API key) |
| 8 | fueleconomy.gov vehicle table | MPG / kWh per 100 mi by make-model-year → energy cost per mile | https://www.fueleconomy.gov/feg/download.shtml | `download_data.py mpg` |

## Optional

| Dataset | Use | Link |
|---|---|---|
| Chicago TNP – Drivers (`j6wf-834c`) | active drivers per month (supply side) | https://data.cityofchicago.org/d/j6wf-834c |
| AFDC charging stations (NREL) | public fast-charger count in Chicago, a charging constraint | https://afdc.energy.gov/stations |
| Vehicle purchase prices (KBB / Edmunds, a few models) | first-stage capital cost | manual |

## Known issues

- TNP data is reported quarterly, so the newest quarter may be missing. The 2026 shock (from March 2026) may be only partly covered.
- Q2 2018 vehicle records were filed as a quarter and put in June 2018.
- Vehicle records often list the model only (e.g. "CAMRY"), so gas vs hybrid versions of the same nameplate can't be told apart. Step 1 reports hybrid share as a range for this reason.
- Fares are rounded to $2.50 and trip times to 15 minutes for privacy.
