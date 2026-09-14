"""
nbp_data.py  -  load the model workfile (3rd sheet) for the NBP Simulator.

Maps columns by position of the 3rd sheet, which is laid out as:
  date, CPI, GC, ET, RXD, RXEURO, GDP, GB, RCB, GI, GOV, CUMOD, YHAT, LS,
  UPILO, ER, COVID, CPI_EA, RXD_EA, GDP_EA, COVID_EA, WPO, DUM20Q2

Exogenous variables (RCB, LS, COVID, COVID_EA, DUM20Q2) already run to 2055Q4.
"""

import numpy as np
import pandas as pd
import nbp_engine as E

GAME_END = "2055Q4"

SHEET3_COLS = ["date", "CPI", "GC", "ET", "RXD", "RXEURO", "GDP", "GB", "RCB",
               "GI", "GOV", "CUMOD", "YHAT", "LS", "UPILO", "ER", "COVID",
               "CPI_EA", "RXD_EA", "GDP_EA", "COVID_EA", "WPO", "DUM20Q2",
               # --- inflation breakdown (new columns) ---
               "CPIFU_WEIGHT", "CPIFU_WEIGHT_DUP", "CPICORE", "CPIFD_WEIGHT",
               "CPIFD", "CPIFU", "CPICORE_EA"]


def _parse_dates(series):
    out = []
    for v in series:
        s = str(v).strip().upper().replace(" ", "")
        if "Q" in s:
            y, q = s.split("Q")
            try:
                out.append(pd.Period(f"{int(y)}Q{int(q)}", freq="Q"))
            except Exception:
                out.append(pd.NaT)
        else:
            out.append(pd.NaT)
    return out


def load_workfile(xlsx_path, sheet=2):
    body = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, skiprows=2)
    body = body.iloc[:, :len(SHEET3_COLS)]
    body.columns = SHEET3_COLS

    dates = _parse_dates(body["date"].tolist())
    df = pd.DataFrame({"date": dates})
    for c in SHEET3_COLS[1:]:
        df[c] = pd.to_numeric(body[c], errors="coerce").values
    df = df[~df["date"].isna()].reset_index(drop=True).set_index("date")

    full_idx = pd.period_range(df.index.min(), GAME_END, freq="Q")
    df = df.reindex(full_idx)

    V = {v: np.full(len(full_idx), np.nan) for v in E.ALL_VARS}
    for v in E.ALL_VARS:
        if v in df.columns:
            V[v] = df[v].astype(float).values.copy()

    hist_end = int(np.max(np.where(~np.isnan(V["GDP"]))))
    if np.all(np.isnan(V["ET"][:hist_end + 1])):
        V["ET"] = V["LS"] * (1 - V["UPILO"] / 100)
    if np.all(np.isnan(V["CUMOD"][:hist_end + 1])):
        V["CUMOD"] = (V["GDP"] - V["YHAT"]) / V["YHAT"] * 100

    # ensure exogenous run to the end; fall back gracefully if any gaps remain
    for d in ["COVID", "COVID_EA", "DUM20Q2"]:
        V[d] = np.where(np.isnan(V[d]), 0.0, V[d])
    # CPI-basket weights are exogenous and run to 2055Q4; forward-fill any gaps
    for w in ["CPIFU_WEIGHT", "CPIFD_WEIGHT"]:
        arr = V[w]
        last = np.nan
        for t in range(len(arr)):
            if np.isnan(arr[t]):
                arr[t] = last
            else:
                last = arr[t]
        # back-fill any leading gaps with the first valid value
        first = next((x for x in arr if not np.isnan(x)), 0.0)
        V[w] = np.where(np.isnan(arr), first, arr)
    last_rcb = int(np.max(np.where(~np.isnan(V["RCB"]))))
    for t in range(last_rcb + 1, len(full_idx)):
        if np.isnan(V["RCB"][t]):
            V["RCB"][t] = V["RCB"][last_rcb]
    ls = V["LS"]
    if np.any(np.isnan(ls[hist_end + 1:])):
        good = ls[~np.isnan(ls)]
        g = np.nanmean(np.diff(np.log(good))) if len(good) > 4 else 0.0
        for t in range(1, len(ls)):
            if np.isnan(ls[t]):
                ls[t] = ls[t - 1] * np.exp(g)

    return V, list(full_idx), hist_end
