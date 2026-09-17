"""
NBP Simulator  -  an interactive monetary-policy game (Streamlit).
Run with:   streamlit run nbp_simulator.py
Keep "Model of Polish economy.xlsx" in the same folder (or use the sidebar
uploader). The game reads the 3rd sheet.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import nbp_engine as E
import nbp_data as D
DATA_PATH = "Model of Polish economy.xlsx"
SHEET     = 2
TARGET    = E.INFLATION_TARGET      # 2.5%
BAND      = 1.0                     # +/- 1 pp tolerance band
MAIN_HIST = 20                      # history quarters in the two main charts
FCAST     = 12                      # forecast quarters
SMALL_WIN = 20                      # history quarters in the small charts
GAP_WIN   = 20                      # quarters in the HP-filter panel
RATE_HIST = 20                      # history quarters in the rate mini-chart
st.set_page_config(page_title="NBP Simulator", layout="wide")
# ---------------------------------------------------------------------------
def qlabel(p):
    return f"{p.year}Q{p.quarter}"
def sparse_ticks(xvals, angle=0, size=8):
    """Show only every 4th date on a categorical x-axis (keeping the last)."""
    xv = list(xvals)
    keep = xv[::-1][::4][::-1]          # every 4th label, latest always kept
    return dict(tickmode="array", tickvals=keep, tickangle=angle,
                tickfont=dict(size=size))
def _d(x, t, k=1):
    """log-difference log(x[t]) - log(x[t-k])."""
    return float(np.log(x[t]) - np.log(x[t - k]))
def _cpi_contribs(V, t):
    """Additive contributions to quarterly core inflation (dlog(CPICORE)) from
    each equation term, plus the energy and food pass-through. The output-gap
    term (c3*cumod) is deliberately excluded. Headline CPI is now a weighted
    identity of energy/food/core, so its drivers are best read off the core
    equation together with the energy and food components it embeds."""
    c = E.COEF["CPICORE"]
    ulc = (np.log(V["ER"][t] * V["ET"][t] / V["GDP"][t])
           - np.log(V["ER"][t-1] * V["ET"][t-1] / V["GDP"][t-1]))
    return {
        "core inflation momentum":        c[0] * _d(V["CPICORE"], t - 1),
        "euro-area core inflation":       c[1] * _d(V["CPICORE_EA"], t),
        "unit labour costs":              c[3] * ulc,
        "core inflation persistence":     c[4] * _d(V["CPICORE"], t - 4),
        "the lagged złoty exchange rate": c[5] * _d(V["RXEURO"], t - 1),
        "energy prices":                  c[6] * _d(V["CPIFU"], t),
        "food prices":                    c[7] * _d(V["CPIFD"], t),
    }
def _headline_component_contribs(V, t):
    """Contribution of each CPI component (energy/food/core) to the
    quarter-on-quarter CHANGE in headline y/y inflation. A component's share of
    headline y/y inflation is approximately weight * component_yoy, so its
    contribution to the *change* is weight * Δ(component_yoy)."""
    yfu = E.yoy(V["CPIFU"]); yfd = E.yoy(V["CPIFD"]); ycore = E.yoy(V["CPICORE"])
    wfu = V["CPIFU_WEIGHT"][t]; wfd = V["CPIFD_WEIGHT"][t]
    w_energy = wfu
    w_food = wfd * (1.0 - wfu)
    w_core = 1.0 - wfu - wfd * (1.0 - wfu)
    return {
        "energy": w_energy * (yfu[t] - yfu[t-1]),
        "food":   w_food   * (yfd[t] - yfd[t-1]),
        "core":   w_core   * (ycore[t] - ycore[t-1]),
    }
def _energy_contribs(V, t):
    """Additive contributions to quarterly energy inflation (dlog(CPIFU))."""
    c = E.COEF["CPIFU"]
    oilfx = (np.log(V["WPO"][t]*V["RXEURO"][t]/V["RXD_EA"][t])
             - np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1]))
    logfx1 = np.log(V["WPO"][t-1]*V["RXEURO"][t-1]/V["RXD_EA"][t-1])
    logulc1 = np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1])
    ec = np.log(V["CPIFU"][t-1]) - c[4]*logfx1 - c[5]*logulc1
    return {
        "global energy and fuel costs": c[2] * oilfx,
        "energy price momentum":        c[1] * _d(V["CPIFU"], t - 1),
        "catch-up to underlying costs": c[3] * ec,
    }
def _food_contribs(V, t):
    """Additive contributions to quarterly food inflation (dlog(CPIFD)).
    The output-gap term (c0*cumod) is deliberately excluded."""
    c = E.COEF["CPIFD"]
    ulc = (np.log(V["ER"][t] * V["ET"][t] / V["GDP"][t])
           - np.log(V["ER"][t-1] * V["ET"][t-1] / V["GDP"][t-1]))
    logulc1 = np.log(V["ER"][t-1]*V["ET"][t-1]/V["GDP"][t-1])
    ec = np.log(V["CPIFD"][t-1]) - logulc1
    return {
        "unit labour costs":              c[1] * ulc,
        "food price persistence":         (c[2]*_d(V["CPIFD"], t-4)
                                           + c[3]*_d(V["CPIFD"], t-8)
                                           + c[4]*_d(V["CPIFD"], t-12)),
        "the lagged złoty exchange rate": c[5] * _d(V["RXEURO"], t - 1),
        "energy pass-through":            c[6] * _d(V["CPIFU"], t),
        "catch-up to underlying costs":   c[7] * ec,
    }
def _gdp_contribs(V, t):
    """Additive contributions to quarterly dlog(GDP) from each equation term.
    Constant, COVID dummy and the lagged output-gap term are excluded."""
    c = E.COEF["GDP"]
    erce = (np.log(V["ER"][t] / V["CPI"][t] * V["ET"][t])
            - np.log(V["ER"][t-1] / V["CPI"][t-1] * V["ET"][t-1]))
    return {
        "real household incomes": c[1] * erce,
        "government spending":    c[2] * _d(V["GOV"], t),
        "euro-area demand":       c[3] * _d(V["GDP_EA"], t),
        "monetary policy":        c[4] * V["RCB"][t],
        "the exchange rate":      c[5] * _d(V["RXEURO"], t),
        "the fiscal stance":      c[7] * (V["GB"][t] - V["GB"][t-1]),
    }
def _contrib_deltas(cfun, V, t):
    """Signed quarter-on-quarter change in each term's contribution."""
    now, prev = cfun(V, t), cfun(V, t - 1)
    return {k: now[k] - prev[k] for k in now}
def _directional_drivers(cfun, V, t, direction, up_lead, up_word,
                         down_lead, down_word):
    """Return a driver phrase whose selection matches the indicator's move.
      * direction > 0 : the two factors that pushed HARDEST up
      * direction < 0 : the two factors that pulled HARDEST down
      * direction == 0: the single biggest up factor 'largely offset by' the
                        single biggest down factor
    """
    delta = _contrib_deltas(cfun, V, t)
    ups   = sorted((k for k in delta if delta[k] > 0),
                   key=lambda k: delta[k], reverse=True)
    downs = sorted((k for k in delta if delta[k] < 0),
                   key=lambda k: delta[k])           # most negative first
    if direction > 0:
        picks = ups[:2] or sorted(delta, key=lambda k: delta[k], reverse=True)[:2]
        return f"{up_lead} {' and '.join(picks)}"
    if direction < 0:
        picks = downs[:2] or sorted(delta, key=lambda k: delta[k])[:2]
        return f"{down_lead} {' and '.join(picks)}"
    # broadly steady -> one up largely offset by one down
    up  = ups[0]   if ups   else max(delta, key=lambda k: delta[k])
    dn  = downs[0] if downs else min(delta, key=lambda k: delta[k])
    return f"{up_word} {up} was largely offset by {down_word} {dn}"
def quarter_narrative(V, dates, li):
    """Rule-based 2-3 sentence briefing on the most recent realised quarter.
    Drivers are selected by the largest quarter-on-quarter change in each
    term's model-implied contribution (no output-gap references)."""
    cpi = E.yoy(V["CPI"]);  gdp = E.yoy(V["GDP"])
    q = qlabel(dates[li])
    cpi_now, cpi_prev = cpi[li], cpi[li-1]
    gdp_now, gdp_prev = gdp[li], gdp[li-1]
    d_cpi = cpi_now - cpi_prev
    d_gdp = gdp_now - gdp_prev
    d_rate = V["RCB"][li] - V["RCB"][li-1]
    # --- inflation sentence ---
    infl_dir = ("rose to" if d_cpi > 0.1 else
                "eased to" if d_cpi < -0.1 else "held broadly steady at")
    if cpi_now > TARGET + BAND:
        stance = "above the tolerance band"
    elif cpi_now < TARGET - BAND:
        stance = "below the tolerance band"
    else:
        stance = "within the tolerance band"
    infl_move = 0 if abs(d_cpi) <= 0.1 else d_cpi
    # which component (energy / food / core) moved headline inflation the most?
    comp = _headline_component_contribs(V, li)
    clabel = {"energy": "energy prices", "food": "food prices",
              "core": "core inflation"}
    if infl_move == 0:
        # broadly steady: biggest up component largely offset by biggest down
        up = max(comp, key=lambda k: comp[k])
        dn = min(comp, key=lambda k: comp[k])
        if comp[up] > 0 and comp[dn] < 0:
            comp_phrase = (f"with {clabel[up]} (pushing up) largely offset by "
                           f"{clabel[dn]} (pulling down)")
        else:
            comp_phrase = "with little net change across energy, food and core"
        key = max(comp, key=lambda k: abs(comp[k]))
    else:
        # components pushing headline in the SAME direction are the drivers
        same = sorted((k for k in comp if (comp[k] > 0) == (infl_move > 0)),
                      key=lambda k: abs(comp[k]), reverse=True)
        same = same or sorted(comp, key=lambda k: abs(comp[k]), reverse=True)
        verb = "pushed up by" if infl_move > 0 else "pulled down by"
        if len(same) >= 2 and abs(comp[same[1]]) > 0.3 * abs(comp[same[0]]):
            comp_phrase = f"{verb} {clabel[same[0]]} and {clabel[same[1]]}"
        else:
            comp_phrase = f"{verb} {clabel[same[0]]}"
        key = same[0]
    s1 = (f"Inflation {infl_dir} **{cpi_now:.1f}%** in {q}, {stance} around "
          f"the {TARGET:.1f}% target, {comp_phrase}.")
    # optional: underlying drivers of the key moving component
    keymove = comp[key]
    if abs(keymove) > 0.05:
        cfun = {"energy": _energy_contribs, "food": _food_contribs,
                "core": _cpi_contribs}[key]
        knoun = {"energy": "The energy move", "food": "The food move",
                 "core": "The core move"}[key]
        drv = _directional_drivers(
            cfun, V, li, keymove,
            up_lead="reflected upward pressure from",
            down_lead="reflected easing in",
            up_word="upward pressure from", down_word="a drag from")
        s1 = s1 + f" {knoun} {drv}."
    # --- growth sentence ---
    grow_dir = ("accelerated to" if d_gdp > 0.2 else
                "slowed to" if d_gdp < -0.2 else "held steady at")
    pace = ("a robust" if gdp_now > 3.5 else
            "a sluggish" if gdp_now < 1.5 else "a moderate")
    gdp_move = 0 if abs(d_gdp) <= 0.2 else d_gdp
    gdp_phrase = _directional_drivers(
        _gdp_contribs, V, li, gdp_move,
        up_lead="supported by stronger",
        down_lead="held back by weaker",
        up_word="a boost from", down_word="a drag from")
    s2 = (f"Real GDP growth {grow_dir} **{gdp_now:.1f}%**, {pace} pace, "
          f"{gdp_phrase}.")
    # --- optional policy sentence (only when the rate moved) ---
    parts = [s1, s2]
    if abs(d_rate) > 0.01:
        move = "hiked" if d_rate > 0 else "cut"
        lean = ("leaning restrictive" if V["RCB"][li] - cpi_now > 0
                else "still accommodative")
        parts.append(f"Having {move} the reference rate by "
                      f"**{abs(d_rate):.2f} pp** to **{V['RCB'][li]:.2f}%**, "
                      f"policy is {lean} relative to current inflation.")
    return " ".join(parts)
def load_state(source, start_label=None):
    V, dates, hist_end = D.load_workfile(source, sheet=SHEET)
    ss = st.session_state
    ss.V, ss.dates, ss.hist_end = V, dates, hist_end
    # pristine copy of the actual data: used both to extract the real historical
    # shocks and as the "actual history" benchmark in the comparison report.
    ss.V0 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in V.items()}
    ss.end_idx = dates.index(pd.Period(D.GAME_END, freq="Q"))
    # start quarter: default = first forecast quarter (pure-forecast, as before)
    if start_label is None:
        start = hist_end + 1
    else:
        start = dates.index(pd.Period(start_label, freq="Q"))
    ss.start = start
    ss.t_cur = start
    ss.start_label = start_label
    seed = int.from_bytes(os.urandom(4), "little") # fresh entropy each game
    ss.seed = seed # keep it so a run is reproducible
    ss.rng = np.random.default_rng(seed)
    ss.rate_input = float(round(V["RCB"][start - 1] * 4) / 4)
    ss.game_over = False
    ss.log = []
def advance_quarter():
    ss = st.session_state
    if ss.game_over:
        return
    t = ss.t_cur
    ss.V["RCB"][t] = ss.rate_input
    if t <= ss.hist_end:
        # historical replay: impose the ACTUAL shocks that hit the economy
        shocks = E.actual_shocks(ss.V0, t)
        period = "hist"
    else:
        # forecast: random shocks
        shocks = E.draw_shocks(ss.rng)
        period = "fcast"
    E.solve_period(ss.V, t, shocks)
    ss.log.append({"quarter": qlabel(ss.dates[t]), "RCB": ss.rate_input,
                   "CPI_yoy": E.yoy(ss.V["CPI"])[t],
                   "GDP_yoy": E.yoy(ss.V["GDP"])[t], "gap": ss.V["CUMOD"][t],
                   "t": t, "period": period})
    ss.t_cur += 1
    if ss.t_cur > ss.end_idx:
        ss.game_over = True
    else:
        ss.rate_input = float(ss.V["RCB"][ss.t_cur - 1])
def end_game():
    st.session_state.game_over = True
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Setup")
up = st.sidebar.file_uploader("Upload workfile (.xlsx, 3rd sheet)", type=["xlsx"])
START_MIN, START_MAX = "2007Q2", "2026Q3"
start_choices = [f"{p.year}Q{p.quarter}"
                 for p in pd.period_range(START_MIN, START_MAX, freq="Q")]
start_label = st.sidebar.selectbox(
    "🕰️ Start quarter", start_choices, index=len(start_choices) - 1,
    help="Pick an earlier quarter to REPLAY history: the actual shocks that hit "
         "the economy are imposed and you choose the interest-rate path. From "
         "2026Q3 the game switches to random-shock forecasting. 2026Q3 = the "
         "classic pure-forecast game.")
src = up if up is not None else DATA_PATH
def _ensure_loaded():
    ss = st.session_state
    need = ("V" not in ss) or (ss.get("start_label") != start_label)
    if need:
        if up is None and not os.path.exists(DATA_PATH):
            st.error("Put 'Model of Polish economy.xlsx' next to this script, "
                     "or upload it in the sidebar.")
            st.stop()
        load_state(src, start_label)
_ensure_loaded()
if st.sidebar.button("🔄 Restart game"):
    load_state(src, start_label)
ss = st.session_state
V, dates = ss.V, ss.dates
t_cur, start, end_idx, hist_end = ss.t_cur, ss.start, ss.end_idx, ss.hist_end
in_history = t_cur <= hist_end        # currently deciding a historical quarter
# ------------------------- welcome & mandate -------------------------------
st.title("🏦 NBP Simulator")
st.markdown(
    f"""
**Welcome, Governor.** You chair the **Monetary Policy Council of Narodowy Bank
Polski**. Every quarter you set the **NBP reference rate** with the stepper
below and click **“Set rate → next quarter.”

**Your mandate:**
- **Primary goal — price stability:** keep **CPI inflation at the {TARGET:.1f}%
  target** (tolerance band **±{BAND:.0f} pp**, i.e. {TARGET-BAND:.1f}–{TARGET+BAND:.1f}%).
- **Secondary goal:** *without prejudice to price stability,* support the
  government’s economic policy, i.e. support GDP growth if inflation is under control.

**Two ways to play** — pick a **start quarter** in the sidebar:
- **Historical replay (2007Q2–2026Q2):** the **actual shocks** that hit the
  economy are imposed; only *your* rate path differs from history. When you
  reach **2026Q3** (or stop earlier) you get a **scorecard vs. the actual MPC**.
- **Forecast (from 2026Q3):** the economy runs on **random shocks**, so every
  playthrough is different, out to **{D.GAME_END}**.

You may stop early with **“End game.”** Your score is the **average absolute
deviation of inflation from {TARGET:.1f}%** and the **average output gap** —
reported separately for the historical and forecast phases.
"""
)
# ------------------------- controls & scoreboard ---------------------------
c1, c2, c3 = st.columns([1.0, 1.3, 1.4])
with c1:
    if not ss.game_over:
        st.subheader(f"🗳️ Decide: **{qlabel(dates[t_cur])}**")
        st.number_input("NBP reference rate, %", min_value=0.0,
                        step=0.25, format="%.2f",
                        key="rate_input",
                        help="Use ▲▼ to move in 0.25 pp steps. "
                             "Default = last quarter’s rate.")
        prev = float(V["RCB"][t_cur - 1])
        st.caption(f"Previous quarter: **{prev:.2f}%** → change "
                   f"**{ss.rate_input - prev:+.2f} pp**")
        st.button("✅ Set rate → next quarter", type="primary",
                  on_click=advance_quarter, use_container_width=True)
        st.button("🏁 End game now", on_click=end_game,
                  use_container_width=True)
    else:
        st.subheader("🏁 Game over")
        st.button("🔄 Play again",
                  on_click=lambda: load_state(up if up is not None else DATA_PATH),
                  use_container_width=True)
with c2:
    if not ss.game_over:
        # --- mini chart: NBP reference rate, 20q history + current choice ---
        r_hist = list(range(max(0, t_cur - RATE_HIST), t_cur))
        xr = [qlabel(dates[i]) for i in r_hist]
        yr = [float(V["RCB"][i]) for i in r_hist]
        xcur = qlabel(dates[t_cur])
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(
            x=xr, y=yr, mode="lines+markers", name="Historical",
            line=dict(color="#1f4e8c", width=2),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        if xr:
            fig_rate.add_trace(go.Scatter(
                x=[xr[-1], xcur], y=[yr[-1], ss.rate_input],
                mode="lines+markers", name="Your choice",
                line=dict(color="#d1495b", width=2, dash="dash"),
                marker=dict(size=11, symbol="diamond", color="#d1495b"),
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        # during historical replay, overlay the actual MPC rate path as a guide
        if in_history:
            r_all = list(range(max(0, t_cur - RATE_HIST), t_cur + 1))
            fig_rate.add_trace(go.Scatter(
                x=[qlabel(dates[i]) for i in r_all],
                y=[float(ss.V0["RCB"][i]) for i in r_all],
                mode="lines", name="Actual (history)",
                line=dict(color="#8a8f98", width=1.5, dash="dot"),
                hovertemplate="%{x}: %{y:.2f}% (actual)<extra></extra>"))
            fig_rate.update_layout(showlegend=True,
                                   legend=dict(orientation="h", y=-0.25,
                                               font=dict(size=8)))
        fig_rate.update_layout(
            title=dict(text="NBP reference rate (RCB, %) - 20q history + your pick",
                       font=dict(size=12)),
            height=230, margin=dict(l=6, r=6, t=32, b=6), showlegend=False,
            xaxis=sparse_ticks(xr + [xcur], size=8),
            yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig_rate, use_container_width=True)
with c3:
    if ss.log:
        last = ss.log[-1]
        st.metric("Latest CPI inflation (y/y)", f"{last['CPI_yoy']:.2f} %",
                  delta=f"{last['CPI_yoy']-TARGET:+.2f} vs target",
                  delta_color="off")
        st.metric("Latest GDP growth (y/y)", f"{last['GDP_yoy']:.2f} %")
        st.progress(min(1.0, (t_cur - start) / max(1, end_idx - start + 1)),
                    text=f"Progress to {D.GAME_END}")
        st.metric("Quarters played", f"{len(ss.log)}")
        st.caption("🎯 In band of deviations from target" if abs(last['CPI_yoy']-TARGET) <= BAND
                   else "⚠️ Outside band of deviations")
    else:
        st.info("No quarters played yet — set your first rate.")
# ------------------------- governor's briefing -----------------------------
if ss.log:
    li = t_cur - 1          # last realised quarter index
    st.markdown(f"#### 📰 Governor’s briefing — {qlabel(dates[li])}")
    st.info(quarter_narrative(V, dates, li))
# ------------------- historical scorecard (player vs. actual) --------------
hist_log = [r for r in ss.log if r["period"] == "hist"]
show_scorecard = hist_log and (t_cur > hist_end or ss.game_over)
if show_scorecard:
    ts = [r["t"] for r in hist_log]
    cpi_act = E.yoy(ss.V0["CPI"])
    gap_act = (ss.V0["GDP"] - ss.V0["YHAT"]) / ss.V0["YHAT"] * 100
    p_infl = np.mean([abs(r["CPI_yoy"] - TARGET) for r in hist_log])
    p_gap = np.mean([r["gap"] for r in hist_log])
    a_infl = np.mean([abs(cpi_act[t] - TARGET) for t in ts])
    a_gap = np.mean([gap_act[t] for t in ts])
    st.markdown("---")
    st.markdown("## 📜 Historical scorecard — your policy vs. what actually happened")
    st.caption(f"Replay period **{hist_log[0]['quarter']}–{hist_log[-1]['quarter']}** "
               f"({len(hist_log)} quarters, actual shocks imposed). Lower inflation "
               f"deviation is better; the output gap is best near zero.")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("#### Avg \\|inflation − 2.5%\\|")
        st.metric("You", f"{p_infl:.2f} pp",
                  delta=f"{p_infl - a_infl:+.2f} pp vs actual",
                  delta_color="inverse")
        st.metric("Actual (history)", f"{a_infl:.2f} pp")
    with sc2:
        st.markdown("#### Avg output gap")
        st.metric("You", f"{p_gap:+.2f} %",
                  delta=f"{p_gap - a_gap:+.2f} pp vs actual", delta_color="off")
        st.metric("Actual (history)", f"{a_gap:+.2f} %")
    if p_infl < a_infl - 0.05:
        st.success("🎉 You kept inflation closer to target than the MPC actually did.")
    elif p_infl > a_infl + 0.05:
        st.warning("The actual MPC kept inflation closer to target than your policy did.")
    else:
        st.info("You broadly matched the actual inflation record.")
    if t_cur > hist_end and not ss.game_over:
        st.caption("▶️ The economy has now reached **2026Q3** — from here the game "
                   "switches to **random-shock forecasting**. Keep setting the rate; "
                   "a separate forecast-phase report follows when you finish.")
    st.markdown("---")
# ------------------------- forward projection ------------------------------
hold_rate = ss.rate_input if not ss.game_over else float(V["RCB"][t_cur - 1])
proj = E.project(V, t_cur, FCAST, hold_rate) if t_cur <= end_idx else \
       {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in V.items()}
gdp_r, cpi_r = E.yoy(V["GDP"]), E.yoy(V["CPI"])
gdp_f, cpi_f = E.yoy(proj["GDP"]), E.yoy(proj["CPI"])
core_r, core_f = E.yoy(V["CPICORE"]), E.yoy(proj["CPICORE"])
# actual (published) history, for comparison during a historical replay
cpi_a, gdp_a = E.yoy(ss.V0["CPI"]), E.yoy(ss.V0["GDP"])
# only meaningful when the game started in the past: in a pure-forecast game
# the realised path IS the actual data, so the extra line would be redundant.
show_actual = ss.start <= hist_end
hist_range = list(range(max(0, t_cur - MAIN_HIST), t_cur))
fc_range = list(range(t_cur, min(t_cur + FCAST, end_idx + 1)))
def main_chart(r, f, title, actual=None, actual_name="Actual (history)"):
    fig = go.Figure()
    xr = [qlabel(dates[i]) for i in hist_range]
    xf = [qlabel(dates[i]) for i in fc_range]
    fig.add_trace(go.Scatter(x=xr, y=[r[i] for i in hist_range],
                  mode="lines+markers", name="Realised",
                  line=dict(color="#1f4e8c", width=3)))
    if fc_range and hist_range:
        fig.add_trace(go.Scatter(x=[xr[-1]] + xf,
                      y=[r[hist_range[-1]]] + [f[i] for i in fc_range],
                      mode="lines+markers", name="Forecast (rate held)",
                      line=dict(color="#d1495b", width=2, dash="dash")))
    # during a historical replay, overlay what ACTUALLY happened, so the player
    # can see how their policy path diverges from the realised record.
    if actual is not None and show_actual:
        a_idx = [i for i in hist_range if i <= hist_end]
        if a_idx:
            fig.add_trace(go.Scatter(
                x=[qlabel(dates[i]) for i in a_idx],
                y=[actual[i] for i in a_idx],
                mode="lines", name=actual_name,
                line=dict(color="#8a8f98", width=1.5, dash="dot"),
                hovertemplate="%{x}: %{y:.2f}% (actual)<extra></extra>"))
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      yaxis_title="% y/y", height=360,
                      margin=dict(l=10, r=10, t=40, b=80),
                      legend=dict(orientation="h", yanchor="top", y=-0.28,
                                  x=0.5, xanchor="center"),
                      xaxis=sparse_ticks(xr + xf))
    return fig
st.markdown("### 📈 Inflation & growth — 20q history + 12q forecast")
m1, m2 = st.columns(2)
with m1:
    fig = main_chart(cpi_r, cpi_f, "Headline & core CPI inflation (% y/y)",
                     actual=cpi_a, actual_name="Headline — actual (history)")
    # overlay core inflation (realised + rate-held forecast)
    xr = [qlabel(dates[i]) for i in hist_range]
    xf = [qlabel(dates[i]) for i in fc_range]
    if hist_range:
        fig.add_trace(go.Scatter(
            x=xr, y=[core_r[i] for i in hist_range], mode="lines+markers",
            name="Core — realised", line=dict(color="#7b52ab", width=2.5)))
    if fc_range and hist_range:
        fig.add_trace(go.Scatter(
            x=[xr[-1]] + xf,
            y=[core_r[hist_range[-1]]] + [core_f[i] for i in fc_range],
            mode="lines+markers", name="Core — forecast",
            line=dict(color="#b08fd6", width=2, dash="dot")))
    # rename the headline traces for clarity (match by name, not position, so
    # the optional "actual" trace can't be renamed by mistake)
    for _tr in fig.data:
        if _tr.name == "Realised":
            _tr.name = "Headline — realised"
        elif _tr.name == "Forecast (rate held)":
            _tr.name = "Headline — forecast (rate held)"
    fig.add_hrect(y0=TARGET-BAND, y1=TARGET+BAND, fillcolor="green",
                  opacity=0.08, line_width=0)
    fig.add_hline(y=TARGET, line_dash="dot", line_color="green",
                  annotation_text="target 2.5%")
    st.plotly_chart(fig, use_container_width=True)
with m2:
    st.plotly_chart(main_chart(gdp_r, gdp_f, "Real GDP growth (% y/y)",
                               actual=gdp_a),
                    use_container_width=True)
# ---------- panel: other endogenous variables (last 20 quarters) -----------
st.markdown("### 🧭 Other key macroeconomic variables — last 20 quarters")
SMALL = [
    ("Nominal wage growth (% y/y)", "ER",     "yoy"),
    ("LFS unemployment rate (%)",    "UPILO",  "level"),
    ("Employment (% y/y)",         "ET",     "yoy"),
    ("EUR/PLN",            "RXEURO", "level"),
    ("EUR/USD",   "RXD_EA", "level"),
    ("Gen. gov. balance (% of GDP)",   "GB",     "level"),
    ("Real government spending (% y/y)", "GOV", "yoy"),
    ("Brent oil price (USD/bbl)",  "WPO",    "level"),
    ("Food inflation (% y/y)",     "CPIFD",  "yoy"),
    ("Energy inflation (% y/y)",   "CPIFU",  "yoy"),
    ("Euro area GDP growth (% y/y)",    "GDP_EA", "yoy"),
    ("Euro area core inflation (% y/y)", "CPICORE_EA", "yoy"),
]
srange = list(range(max(0, t_cur - SMALL_WIN), t_cur))
xs = [qlabel(dates[i]) for i in srange]
cols = st.columns(5)
for k, (label, var, tr) in enumerate(SMALL):
    series = E.yoy(V[var]) if tr == "yoy" else V[var]
    with cols[k % 5]:
        f = go.Figure(go.Scatter(x=xs, y=[series[i] for i in srange],
                      mode="lines+markers", line=dict(color="#1f4e8c", width=2)))
        f.update_layout(title=dict(text=label, font=dict(size=11)),
                        height=210, margin=dict(l=6, r=6, t=34, b=6),
                        xaxis=sparse_ticks(xs, angle=0, size=7),
                        yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(f, use_container_width=True)
# ---------- panel: potential vs actual GROWTH (HP filter) + policy IRF ------
st.markdown("### 🪚 HP-filtered potential growth, output gap, and monetary-policy impact")
gap, pot = E.hp_gap(V["GDP"][:t_cur])
pot_yoy = E.yoy(pot)
grange = list(range(max(0, t_cur - GAP_WIN), t_cur))
xg = [qlabel(dates[i]) for i in grange]
p1, p2, p3 = st.columns(3)
with p1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=xg, y=[gdp_r[i] for i in grange], name="Actual GDP",
                mode="lines+markers", line=dict(color="#1f4e8c", width=3)))
    f.add_trace(go.Scatter(x=xg, y=[pot_yoy[i] for i in grange],
                name="Potential (HP)", mode="lines+markers",
                line=dict(color="#e08a1e", width=2, dash="dash")))
    f.update_layout(title=dict(text="Actual vs HP-filtered potential growth (% y/y)",
                               x=0.5, xanchor="center"),
                    height=340, margin=dict(l=10, r=10, t=40, b=80),
                    legend=dict(orientation="h", yanchor="top", y=-0.28,
                                x=0.5, xanchor="center"),
                    xaxis=sparse_ticks(xg), yaxis_title="% y/y")
    st.plotly_chart(f, use_container_width=True)
with p2:
    f = go.Figure(go.Bar(x=xg, y=[gap[i] for i in grange],
                  marker_color=["#2a9d8f" if gap[i] >= 0 else "#d1495b"
                                for i in grange]))
    f.add_hline(y=0, line_color="black")
    f.update_layout(title=dict(text="HP-filtered output gap (% of potential)",
                               x=0.5, xanchor="center"),
                    height=340, margin=dict(l=10, r=10, t=40, b=80),
                    xaxis=sparse_ticks(xg), yaxis_title="%")
    st.plotly_chart(f, use_container_width=True)
with p3:
    # impulse response to a PERMANENT +100 bp hike, from the current state:
    # deviation of headline CPI and GDP growth vs. a rate-held baseline.
    IRF_H = 13                                   # quarters 0..12
    r_base = ss.rate_input if not ss.game_over else float(V["RCB"][t_cur - 1])
    base_p = E.project(V, t_cur, IRF_H, r_base)
    hike_p = E.project(V, t_cur, IRF_H, r_base + 1.0)
    cpi_b, cpi_s = E.yoy(base_p["CPI"]), E.yoy(hike_p["CPI"])
    gdp_b, gdp_s = E.yoy(base_p["GDP"]), E.yoy(hike_p["GDP"])
    n_ir = min(IRF_H, end_idx + 1 - t_cur)
    hx = list(range(n_ir))
    d_cpi = [cpi_s[t_cur + k] - cpi_b[t_cur + k] for k in hx]
    d_gdp = [gdp_s[t_cur + k] - gdp_b[t_cur + k] for k in hx]
    f = go.Figure()
    f.add_hline(y=0, line_color="black", line_width=1)
    f.add_trace(go.Scatter(x=hx, y=d_cpi, mode="lines+markers",
                name="Headline CPI", line=dict(color="#1f4e8c", width=2.5),
                hovertemplate="q%{x}: %{y:.2f} pp<extra></extra>"))
    f.add_trace(go.Scatter(x=hx, y=d_gdp, mode="lines+markers",
                name="GDP growth", line=dict(color="#d1495b", width=2.5),
                hovertemplate="q%{x}: %{y:.2f} pp<extra></extra>"))
    f.update_layout(title=dict(text="Impact of a permanent +100 bp hike (y/y, pp)",
                               x=0.5, xanchor="center"),
                    height=340, margin=dict(l=10, r=10, t=40, b=80),
                    legend=dict(orientation="h", yanchor="top", y=-0.28,
                                x=0.5, xanchor="center"),
                    xaxis=dict(title="Quarters since hike", dtick=2,
                               tickmode="linear"),
                    yaxis_title="Deviation from baseline (pp)")
    st.plotly_chart(f, use_container_width=True)
# --------------- end-of-game report (forecast phase, 2026Q3→) --------------
fcast_log = [r for r in ss.log if r["period"] == "fcast"]
if ss.game_over and fcast_log:
    st.markdown("---")
    st.markdown("## 🏆 Final report — forecast phase (2026Q3 →)")
    infl_dev = np.mean([abs(r["CPI_yoy"] - TARGET) for r in fcast_log])
    gap_avg = np.mean([r["gap"] for r in fcast_log])
    r1, r2, r3 = st.columns(3)
    r1.metric("Avg |inflation − 2.5%|", f"{infl_dev:.2f} pp")
    r2.metric("Avg output gap (actual)", f"{gap_avg:+.2f} %")
    r3.metric("Quarters governed",
              f"{len(fcast_log)}  ({fcast_log[0]['quarter']}–{fcast_log[-1]['quarter']})")
    if infl_dev <= 0.5 and abs(gap_avg) <= 1.0:
        st.success("🥇 Outstanding — price stability with a balanced economy.")
    elif infl_dev <= 1.0:
        st.info("🥈 Solid — inflation broadly anchored near target.")
    else:
        st.warning("🥉 Inflation drifted from target — some explaining to do.")
# --------------------------- play-by-play download -------------------------
if ss.game_over and ss.log:
    disp_cols = ["quarter", "period", "RCB", "CPI_yoy", "GDP_yoy", "gap"]
    df_log = pd.DataFrame(ss.log)[disp_cols]
    st.dataframe(df_log.round(2), use_container_width=True, height=260)
    st.download_button("⬇️ Download play-by-play (CSV)",
                       df_log.to_csv(index=False).encode(),
                       file_name="nbp_simulator_game.csv", mime="text/csv")
