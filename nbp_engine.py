"""
nbp_engine.py  -  simulation engine for the NBP Simulator game.
Recreates the simple simultaneous macro-econometric model of the Polish economy
and solves it one quarter at a time with a Gauss-Seidel iteration. Each realised
quarter draws stochastic shocks (~ N(0, equation S.E.)); the forward projection
shown in the charts is deterministic (shocks = 0) with the policy rate held.
NOTE on variable meaning:
  * ER   = nominal wages (gross earnings per employee), NOT an exchange rate.
  * RXEURO = PLN per EUR ;  RXD_EA = USD per EUR (EUR/USD).
"""
import numpy as np
# ---------------------------------------------------------------------------
# coefficients and residual standard errors (from EViews estimation)
# ---------------------------------------------------------------------------
COEF = {
    # --- inflation breakdown: headline CPI is now an identity (weighted sum of
    #     energy [CPIFU], food [CPIFD] and core [CPICORE]); each component and
    #     euro-area core [CPICORE_EA] has its own behavioural equation. ---
    "CPIFU":      [-0.8722417, 0.1198889, 0.1082163, -0.2068935, 0.3265544,
                   1.0844373],
    "CPICORE":    [0.4348811, 0.2711221, 0.0002926, 0.0722825, 0.1176052,
                   0.0187040, 0.0491025, 0.0653273],
    "CPICORE_EA": [-5.202e-05, 0.9476215, 0.0041660, 3.102e-05, -0.0139773],
    "CPIFD":      [0.0012941, 0.3417762, 0.2973576, 0.1982597, 0.1999393,
                   0.1272386, 0.2056404, 0.0018256],
    "ER":     [0.0060742, 0.2297092, 0.1655492, 0.2846569, -0.0344416, 0.0518027],
    "GB":     [9.5423318, 0.5999016, -48.099453, -4.3003161, -7.7162334],
    "GDP":    [0.0077582, 0.1378909, 0.0979589, 0.5526196, -0.0006144,
               0.0370731, -0.0434601, -0.0014051],
    "GDP_EA": [0.0026484, 0.1948505, -0.0008987, -0.1140358, 0.0423129],
    "GOV":    [-0.1372427, 0.0005877],           # + fixed constant 0.008
    "RXEURO": [0.0006654, 0.1658873, -0.0069962],
    "YHAT":   [0.0051753, 1.2078064, -0.3881609, -0.1558138, -0.0001615,
               -1.454e-05, 0.0188316, -0.0124695],
    "RXD_EA": [-0.0007258, 0.3533210, -0.1504215],
    "UPILO":  [0.0448500, 0.6460633, -8.3148628, 20.068489],
    "WPO":    [-12.853889, 1.0982112, -0.1470449, 400.86014, 30.812134,
               79.164554, 12.679196],
}
SE = {
    "CPIFU": 0.01, "CPICORE": 0.0023, "CPICORE_EA": 0.002, "CPIFD": 0.009,
    "ER": 0.007, "GB": 1.1340887,
    "GDP": 0.0045, "GDP_EA": 0.0033, "GOV": 0.0195674, "RXEURO": 0.0327465,
    "YHAT": 0.0017596, "RXD_EA": 0.0359080, "UPILO": 0.23, "WPO": 8.2630845,
}
STOCH_VARS = list(SE.keys())
ALL_VARS = ["CPI", "GC", "ET", "RXD", "RXEURO", "GDP", "GB", "RCB", "GI", "GOV",
            "CUMOD", "YHAT", "LS", "UPILO", "ER", "COVID", "CPI_EA", "RXD_EA",
            "GDP_EA", "COVID_EA", "WPO", "DUM20Q2",
            "CPIFU", "CPIFD", "CPICORE", "CPICORE_EA",
            "CPIFU_WEIGHT", "CPIFD_WEIGHT"]
EXOG_VARS = ["RCB", "LS", "COVID", "COVID_EA", "DUM20Q2",
             "CPIFU_WEIGHT", "CPIFD_WEIGHT"]
INFLATION_TARGET = 2.5
COVID_AR = 0.9      # AR(1) decay of the COVID dummies in the forecast
# ---------------------------------------------------------------------------
def _dl(x, t, k=1):
    return np.log(x[t]) - np.log(x[t - k])
def _dlr(a, b, t):
    return np.log(a[t] / b[t]) - np.log(a[t - 1] / b[t - 1])
def draw_shocks(rng):
    return {v: float(rng.normal(0.0, SE[v])) for v in STOCH_VARS}
def zero_shocks():
    return {v: 0.0 for v in STOCH_VARS}
def actual_shocks(V, t):
    """Back out the structural residuals at quarter t from *actual* data, so a
    historical replay imposes the true shocks that hit the economy (only the
    policy rate differs from history). Mirrors every equation in solve_period
    and solves for the additive/multiplicative residual, using model-consistent
    CUMOD = (GDP-YHAT)/YHAT*100 exactly as solve_period does."""
    s = {}
    c = COEF["RXD_EA"]
    s["RXD_EA"] = (_dl(V["RXD_EA"], t)
                   - (c[0] + c[1]*_dl(V["RXD_EA"], t-1) + c[2]*_dl(V["RXD_EA"], t-2)))
    c = COEF["GDP_EA"]
    s["GDP_EA"] = (_dl(V["GDP_EA"], t)
                   - (c[0] + c[1]*_dl(V["GDP_EA"], t-1)
                      + c[2]*(V["COVID_EA"][t]-V["COVID_EA"][t-1])
                      + c[3]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])
                      + c[4]*V["DUM20Q2"][t]))
    c = COEF["WPO"]
    s["WPO"] = (V["WPO"][t]
                - (c[0] + c[1]*V["WPO"][t-1] + c[2]*V["WPO"][t-2]
                   + c[3]*_dl(V["GDP_EA"], t)
                   + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])
                   + c[5]*_dl(V["RXD_EA"], t) + c[6]*V["RXD_EA"][t]))
    c = COEF["CPICORE_EA"]
    s["CPICORE_EA"] = (_dl(V["CPICORE_EA"], t)
                       - (c[0] + c[1]*_dl(V["CPICORE_EA"], t-4)
                          + c[2]*_dlr(V["WPO"], V["RXD_EA"], t-2)
                          + c[3]*V["COVID_EA"][t] + c[4]*_dl(V["RXD_EA"], t-2)))
    c = COEF["RXEURO"]
    s["RXEURO"] = (_dl(V["RXEURO"], t)
                   - (c[0] + c[1]*_dl(V["RXEURO"], t-1)
                      + c[2]*(V["RCB"][t]-V["RCB"][t-1])))
    c = COEF["GOV"]
    s["GOV"] = (_dl(V["GOV"], t)
                - (0.008 + c[0]*_dl(V["GOV"], t-1)
                   + c[1]*(V["COVID"][t]-V["COVID"][t-1])))
    c = COEF["YHAT"]
    s["YHAT"] = (_dl(V["YHAT"], t)
                 - (c[0] + c[1]*_dl(V["YHAT"], t-1) + c[2]*_dl(V["YHAT"], t-2)
                    + c[3]*_dl(V["YHAT"], t-3) + c[4]*(V["COVID"][t]-V["COVID"][t-1])
                    + c[5]*V["YHAT"][t-1]/V["GDP_EA"][t-1] + c[6]*V["DUM20Q2"][t]
                    + c[7]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])))
    def cum(k):
        return (V["GDP"][k]-V["YHAT"][k])/V["YHAT"][k]*100
    c = COEF["GDP"]
    erce = (np.log(V["ER"][t]/V["CPI"][t]*V["ET"][t])
            - np.log(V["ER"][t-1]/V["CPI"][t-1]*V["ET"][t-1]))
    s["GDP"] = (_dl(V["GDP"], t)
                - (c[0] + c[1]*erce + c[2]*_dl(V["GOV"], t) + c[3]*_dl(V["GDP_EA"], t)
                   + c[4]*V["RCB"][t] + c[5]*_dl(V["RXEURO"], t) + c[6]*V["DUM20Q2"][t]
                   + c[7]*(V["GB"][t]-V["GB"][t-1]) - 0.2*cum(t-1)/100))
    c = COEF["UPILO"]
    s["UPILO"] = (V["UPILO"][t] - V["UPILO"][t-1]
                  - (c[0] + c[1]*(V["UPILO"][t-1]-V["UPILO"][t-2])
                     + c[2]*_dl(V["GDP"], t) + c[3]*_dl(V["LS"], t)))
    # unit labour cost and the energy/oil cost term (shared by components)
    ulc = (np.log(V["ER"][t]*V["ET"][t]/V["GDP"][t])
           - np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1]))
    oilfx = (np.log(V["WPO"][t]*V["RXEURO"][t]/V["RXD_EA"][t])
             - np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1]))
    logfx1 = np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1])
    logulc1 = np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1])
    c = COEF["CPIFU"]
    s["CPIFU"] = (_dl(V["CPIFU"], t)
                  - (c[0] + c[1]*_dl(V["CPIFU"], t-1) + c[2]*oilfx
                     + c[3]*(np.log(V["CPIFU"][t-1]) - c[4]*logfx1 - c[5]*logulc1)))
    c = COEF["CPIFD"]
    s["CPIFD"] = (_dl(V["CPIFD"], t)
                  - (c[0]*cum(t) + c[1]*ulc + c[2]*_dl(V["CPIFD"], t-4)
                     + c[3]*_dl(V["CPIFD"], t-8) + c[4]*_dl(V["CPIFD"], t-12)
                     + c[5]*_dl(V["RXEURO"], t-1) + c[6]*_dl(V["CPIFU"], t)
                     + c[7]*(np.log(V["CPIFD"][t-1]) - logulc1)))
    c = COEF["CPICORE"]
    s["CPICORE"] = (_dl(V["CPICORE"], t)
                    - (c[0]*_dl(V["CPICORE"], t-1) + c[1]*_dl(V["CPICORE_EA"], t)
                       + c[2]*cum(t) + c[3]*ulc + c[4]*_dl(V["CPICORE"], t-4)
                       + c[5]*_dl(V["RXEURO"], t-1) + c[6]*_dl(V["CPIFU"], t)
                       + c[7]*_dl(V["CPIFD"], t)))
    c = COEF["ER"]
    s["ER"] = (_dl(V["ER"], t)
               - (c[0] + c[1]*_dl(V["ER"], t-1) + c[2]*_dl(V["CPI"], t)
                  - 0.01*V["UPILO"][t]/100 + c[3]*_dl(V["ER"], t-2)
                  + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1]) + c[5]*_dl(V["GOV"], t)))
    c = COEF["GB"]
    s["GB"] = (V["GB"][t]
               - (c[0] + c[1]*V["GB"][t-1] + c[2]*V["GOV"][t]/V["GDP"][t]
                  + c[3]*V["DUM20Q2"][t] + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])))
    return s
# ---------------------------------------------------------------------------
ENDO = ["CPI", "CPIFU", "CPIFD", "CPICORE", "CPICORE_EA", "ER", "GB", "GDP",
        "GDP_EA", "GOV", "RXEURO", "YHAT", "RXD_EA", "UPILO", "WPO", "CUMOD",
        "ET"]
def solve_period(V, t, shocks, max_iter=500, tol=1e-11):
    endo = ENDO
    for nm in endo:
        if np.isnan(V[nm][t]):
            V[nm][t] = V[nm][t - 1]
    for _ in range(max_iter):
        old = np.array([V[nm][t] for nm in endo])
        c = COEF["RXD_EA"]
        V["RXD_EA"][t] = V["RXD_EA"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["RXD_EA"], t-1) + c[2]*_dl(V["RXD_EA"], t-2)
            + shocks["RXD_EA"])
        if V["RXD_EA"][t] < 0.05:
            V["RXD_EA"][t] = 0.05
        c = COEF["GDP_EA"]
        V["GDP_EA"][t] = V["GDP_EA"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["GDP_EA"], t-1)
            + c[2]*(V["COVID_EA"][t]-V["COVID_EA"][t-1])
            + c[3]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])
            + c[4]*V["DUM20Q2"][t] + shocks["GDP_EA"])
        c = COEF["WPO"]
        V["WPO"][t] = (c[0] + c[1]*V["WPO"][t-1] + c[2]*V["WPO"][t-2]
                       + c[3]*_dl(V["GDP_EA"], t)
                       + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])
                       + c[5]*_dl(V["RXD_EA"], t) + c[6]*V["RXD_EA"][t]
                       + shocks["WPO"])
        if V["WPO"][t] < 1.0:
            V["WPO"][t] = 1.0
        c = COEF["CPICORE_EA"]
        V["CPICORE_EA"][t] = V["CPICORE_EA"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["CPICORE_EA"], t-4)
            + c[2]*_dlr(V["WPO"], V["RXD_EA"], t-2) + c[3]*V["COVID_EA"][t]
            + c[4]*_dl(V["RXD_EA"], t-2) + shocks["CPICORE_EA"])
        c = COEF["RXEURO"]
        V["RXEURO"][t] = V["RXEURO"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["RXEURO"], t-1) + c[2]*(V["RCB"][t]-V["RCB"][t-1])
            + shocks["RXEURO"])
        c = COEF["GOV"]
        V["GOV"][t] = V["GOV"][t-1] * np.exp(
            0.008 + c[0]*_dl(V["GOV"], t-1) + c[1]*(V["COVID"][t]-V["COVID"][t-1])
            + shocks["GOV"])
        c = COEF["YHAT"]
        V["YHAT"][t] = V["YHAT"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["YHAT"], t-1) + c[2]*_dl(V["YHAT"], t-2)
            + c[3]*_dl(V["YHAT"], t-3) + c[4]*(V["COVID"][t]-V["COVID"][t-1])
            + c[5]*V["YHAT"][t-1]/V["GDP_EA"][t-1] + c[6]*V["DUM20Q2"][t]
            + c[7]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1]) + shocks["YHAT"])
        V["ET"][t] = V["LS"][t] * (1 - V["UPILO"][t]/100)
        c = COEF["GDP"]
        erce = (np.log(V["ER"][t]/V["CPI"][t]*V["ET"][t])
                - np.log(V["ER"][t-1]/V["CPI"][t-1]*V["ET"][t-1]))
        V["GDP"][t] = V["GDP"][t-1] * np.exp(
            c[0] + c[1]*erce + c[2]*_dl(V["GOV"], t) + c[3]*_dl(V["GDP_EA"], t)
            + c[4]*V["RCB"][t] + c[5]*_dl(V["RXEURO"], t) + c[6]*V["DUM20Q2"][t]
            + c[7]*(V["GB"][t]-V["GB"][t-1]) - 0.2*V["CUMOD"][t-1]/100
            + shocks["GDP"])
        V["CUMOD"][t] = (V["GDP"][t]-V["YHAT"][t])/V["YHAT"][t]*100
        c = COEF["UPILO"]
        V["UPILO"][t] = V["UPILO"][t-1] + (
            c[0] + c[1]*(V["UPILO"][t-1]-V["UPILO"][t-2])
            + c[2]*_dl(V["GDP"], t) + c[3]*_dl(V["LS"], t) + shocks["UPILO"])
        V["UPILO"][t] = min(max(V["UPILO"][t], 0.1), 40.0)
        V["ET"][t] = V["LS"][t] * (1 - V["UPILO"][t]/100)
        # ---- inflation breakdown: energy (CPIFU), food (CPIFD), core (CPICORE),
        #      then headline CPI as the weighted identity ----
        ulc = (np.log(V["ER"][t]*V["ET"][t]/V["GDP"][t])
               - np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1]))
        oilfx = (np.log(V["WPO"][t]*V["RXEURO"][t]/V["RXD_EA"][t])
                 - np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1]))
        logfx1 = np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1])
        logulc1 = np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1])
        c = COEF["CPIFU"]
        V["CPIFU"][t] = V["CPIFU"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["CPIFU"], t-1) + c[2]*oilfx
            + c[3]*(np.log(V["CPIFU"][t-1]) - c[4]*logfx1 - c[5]*logulc1)
            + shocks["CPIFU"])
        c = COEF["CPIFD"]
        V["CPIFD"][t] = V["CPIFD"][t-1] * np.exp(
            c[0]*V["CUMOD"][t] + c[1]*ulc + c[2]*_dl(V["CPIFD"], t-4)
            + c[3]*_dl(V["CPIFD"], t-8) + c[4]*_dl(V["CPIFD"], t-12)
            + c[5]*_dl(V["RXEURO"], t-1) + c[6]*_dl(V["CPIFU"], t)
            + c[7]*(np.log(V["CPIFD"][t-1]) - logulc1) + shocks["CPIFD"])
        c = COEF["CPICORE"]
        V["CPICORE"][t] = V["CPICORE"][t-1] * np.exp(
            c[0]*_dl(V["CPICORE"], t-1) + c[1]*_dl(V["CPICORE_EA"], t)
            + c[2]*V["CUMOD"][t] + c[3]*ulc + c[4]*_dl(V["CPICORE"], t-4)
            + c[5]*_dl(V["RXEURO"], t-1) + c[6]*_dl(V["CPIFU"], t)
            + c[7]*_dl(V["CPIFD"], t) + shocks["CPICORE"])
        # headline identity: weighted sum of component log-changes
        wfu = V["CPIFU_WEIGHT"][t]
        wfd = V["CPIFD_WEIGHT"][t]
        wcore = 1.0 - wfu - wfd*(1.0 - wfu)
        V["CPI"][t] = V["CPI"][t-1] * np.exp(
            _dl(V["CPIFU"], t)*wfu + _dl(V["CPIFD"], t)*wfd*(1.0 - wfu)
            + _dl(V["CPICORE"], t)*wcore)
        c = COEF["ER"]
        V["ER"][t] = V["ER"][t-1] * np.exp(
            c[0] + c[1]*_dl(V["ER"], t-1) + c[2]*_dl(V["CPI"], t)
            - 0.01*V["UPILO"][t]/100 + c[3]*_dl(V["ER"], t-2)
            + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1]) + c[5]*_dl(V["GOV"], t)
            + shocks["ER"])
        c = COEF["GB"]
        V["GB"][t] = (c[0] + c[1]*V["GB"][t-1] + c[2]*V["GOV"][t]/V["GDP"][t]
                      + c[3]*V["DUM20Q2"][t] + c[4]*(V["DUM20Q2"][t]-V["DUM20Q2"][t-1])
                      + shocks["GB"])
        new = np.array([V[nm][t] for nm in endo])
        denom = np.where(np.abs(old) < 1e-9, 1.0, old)
        if np.max(np.abs((new-old)/denom)) < tol:
            break
    return V
def project(V, t_start, horizon, rate_hold):
    P = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in V.items()}
    z = zero_shocks()
    n = len(P["GDP"])
    for t in range(t_start, min(t_start + horizon, n)):
        P["RCB"][t] = rate_hold
        # COVID dummies follow an AR(1) with rho = COVID_AR in the forecast: no
        # pandemic foresight before COVID (0 -> 0) and a gradual dying-out
        # afterwards.
        P["COVID"][t] = COVID_AR * P["COVID"][t-1]
        P["COVID_EA"][t] = COVID_AR * P["COVID_EA"][t-1]
        # Zero the one-off 2020Q2 dummy inside the forecast horizon. This ensures
        # a forecast prepared BEFORE 2020Q2 has no COVID foresight (it never
        # "sees" the scheduled 2020Q2=1 spike). For a forecast starting in
        # 2020Q3+, the future DUM20Q2 values are already 0, while the 2020Q2=1
        # value sits in *history* (outside the horizon), so its natural switch
        # back to 0 -- and the associated rebound -- is preserved.
        P["DUM20Q2"][t] = 0.0
        for nm in ENDO:
            P[nm][t] = np.nan
        solve_period(P, t, z)
    return P
def yoy(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    out[4:] = (x[4:]/x[:-4] - 1)*100
    return out
def hp_gap(gdp_level, lamb=1600):
    """HP-filter potential (level) and output gap (%) from a real GDP level."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    g = np.asarray(gdp_level, dtype=float)
    mask = ~np.isnan(g)
    trend = np.full_like(g, np.nan)
    if mask.sum() >= 8:
        _, tr = hpfilter(np.log(g[mask]), lamb=lamb)
        trend[mask] = np.exp(tr)
    gap = (g - trend)/trend*100
    return gap, trend
