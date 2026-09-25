"""
How late was the Fed? Measuring how far behind the curve the Federal Reserve fell in 2021-22.

Benchmarks for where the policy rate "should" have been:
  1. Textbook Taylor (1993) and balanced-approach (1999) rules, plus an inertial variant.
  2. The Fed's own estimated reaction function, 1987-2019 (partial-adjustment OLS).
  3. A gradient-boosting model of the historical Fed, trained 1987-2019 and asked
     "what would the pre-2020 Fed have done with 2021-22 data?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

import fred
from utils import DATA, HIGHLIGHT, MUTED, PALETTE, md_table, savefig, setup, write_results

SRC = "FRED (Federal Reserve Bank of St. Louis): FEDFUNDS, PCEPILFE, PCEPI, UNRATE, NROU, GDPC1, GDPPOT"
TRAIN = ("1987-07-01", "2019-12-31")   # Greenspan to pre-COVID
FOCUS = ("2020-01-01", None)
R_STAR, PI_STAR = 2.0, 2.0


def load() -> pd.DataFrame:
    m = fred.series(["FEDFUNDS", "PCEPILFE", "PCEPI", "UNRATE"])
    q = fred.series(["GDPC1", "GDPPOT", "NROU"])
    mq = m.resample("QS").mean()
    df = mq.join(q, how="left")
    df["core"] = 100 * np.log(df["PCEPILFE"] / df["PCEPILFE"].shift(4))
    df["headline"] = 100 * np.log(df["PCEPI"] / df["PCEPI"].shift(4))
    df["gap"] = 100 * (df["GDPC1"] / df["GDPPOT"] - 1)
    df["ugap"] = df["NROU"] - df["UNRATE"]                         # + = labour market tighter than normal
    df["ff"] = df["FEDFUNDS"]
    df["ff_lag"] = df["ff"].shift(1)
    df["core_chg"] = df["core"] - df["core"].shift(2)
    df = df.dropna(subset=["ff", "core", "gap", "ff_lag"])
    df.to_csv(DATA / "fed_quarterly.csv")
    return df, m["FEDFUNDS"].dropna()


def rules(df: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame(index=df.index)
    r["Taylor (1993)"] = R_STAR + df["core"] + 0.5 * (df["core"] - PI_STAR) + 0.5 * df["gap"]
    r["Balanced approach (1999)"] = R_STAR + df["core"] + 0.5 * (df["core"] - PI_STAR) + 1.0 * df["gap"]
    inert = [df["ff"].iloc[0]]
    for t in range(1, len(df)):                                     # inertial rule, iterated on its own path
        inert.append(0.85 * inert[-1] + 0.15 * r["Taylor (1993)"].iloc[t])
    r["Inertial Taylor"] = inert
    return r.clip(lower=0.0)                                        # the effective lower bound


def main() -> None:
    setup()
    df, ff_monthly = load()
    R = rules(df)
    tr = df.loc[TRAIN[0]:TRAIN[1]]
    md = []

    # ---- Estimated reaction function (partial adjustment) ---------------------------------------
    ols = smf.ols("ff ~ ff_lag + core + gap", data=tr).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rho = ols.params["ff_lag"]
    lr = {k: ols.params[k] / (1 - rho) for k in ["core", "gap"]}
    # dynamic simulation from 2020 on: the rule feeds on its own lagged prediction, not the actual rate
    sim = df["ff"].copy()
    idx = df.index.get_loc(df.loc[FOCUS[0]:].index[0])
    for t in range(idx, len(df)):
        prev = sim.iloc[t - 1]
        sim.iloc[t] = max(0.0, ols.params["Intercept"] + rho * prev + ols.params["core"] * df["core"].iloc[t]
                          + ols.params["gap"] * df["gap"].iloc[t])
    R["Estimated Fed rule (1987–2019)"] = sim.where(df.index >= FOCUS[0])

    # ---- ML version of the historical Fed -----------------------------------------------------------
    feats = ["core", "headline", "gap", "ugap", "core_chg"]
    trm = tr.dropna(subset=feats)
    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                    random_state=0).fit(trm[feats], trm["ff"])
    ml = pd.Series(np.nan, index=df.index)
    ok = df[feats].notna().all(axis=1)
    ml[ok] = gbm.predict(df.loc[ok, feats])
    R["ML Fed (gradient boosting)"] = ml.clip(lower=0).where(df.index >= FOCUS[0])
    cvp = np.full(len(trm), np.nan)
    for a, b in TimeSeriesSplit(5).split(trm):
        cvp[b] = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                           random_state=0).fit(trm[feats].iloc[a], trm["ff"].iloc[a]).predict(trm[feats].iloc[b])
    okc = ~np.isnan(cvp)
    cv_rmse = float(np.sqrt(np.mean((cvp[okc] - trm["ff"].values[okc]) ** 2)))

    # ---- Figure 1: long view ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df.index, df["ff"], color="#111", lw=2.6, label="Actual fed funds rate")
    for (name, c) in zip(["Taylor (1993)", "Balanced approach (1999)", "Inertial Taylor"], PALETTE):
        ax.plot(R.index, R[name], color=c, lw=1.4, alpha=0.9, label=name)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-30"), color=HIGHLIGHT, alpha=0.08)
    ax.set_ylabel("%")
    ax.set_title("The fed funds rate vs textbook policy rules (shaded: 2021–mid 2022)")
    ax.legend(ncol=2, fontsize=9)
    f1 = savefig(fig, "01_rules_long_run", SRC)

    # ---- Figure 2: zoom on 2019+ ---------------------------------------------------------------------
    z = df.loc["2019-01-01":]
    Rz = R.loc["2019-01-01":]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(z.index, z["ff"], color="#111", lw=3, label="Actual fed funds rate")
    cols = {"Taylor (1993)": PALETTE[0], "Balanced approach (1999)": PALETTE[1], "Inertial Taylor": PALETTE[2],
            "Estimated Fed rule (1987–2019)": PALETTE[4], "ML Fed (gradient boosting)": HIGHLIGHT}
    for name, c in cols.items():
        ax.plot(Rz.index, Rz[name], color=c, lw=2, ls="--" if "Fed" in name else "-", label=name)
    ax.plot(z.index, z["core"], color=MUTED, lw=1.5, label="Core PCE inflation (y/y)")
    ax.set_ylabel("%")
    ax.set_title("Behind the curve? Actual policy vs five benchmarks since 2019")
    ax.legend(fontsize=8.5, ncol=2)
    f2 = savefig(fig, "02_behind_the_curve", SRC)

    # ---- Metrics ----------------------------------------------------------------------------------
    win = df.loc["2021-01-01":"2022-12-31"]
    rows = []
    # first month after 2021 with the effective rate at or above 0.15% (the March 2022 hike lifts the monthly average to ~0.2)
    lm = ff_monthly[(ff_monthly.index >= "2021-01-01") & (ff_monthly >= 0.15)]
    liftoff = lm.index.min() if len(lm) else pd.NaT
    for name in cols:
        pr = R.loc[win.index, name]
        gapm = (pr - win["ff"]).mean()
        crossed = R.index[(R.index >= "2020-07-01") & (R[name] >= 0.75)]
        first = crossed.min() if len(crossed) else pd.NaT
        lag_q = (liftoff.to_period("Q") - first.to_period("Q")).n if pd.notna(first) and pd.notna(liftoff) else np.nan
        rows.append({"Benchmark": name, "Avg gap 2021–22 (pp)": gapm, "Peak gap (pp)": (pr - win["ff"]).max(),
                     "Rule says hike (quarter)": str(first.to_period("Q")) if pd.notna(first) else "–",
                     "Quarters late": lag_q})
    tab = pd.DataFrame(rows)

    # Figure 3: bar of avg gap
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.barh(tab["Benchmark"], tab["Avg gap 2021–22 (pp)"], color=[cols[n] for n in tab["Benchmark"]])
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Average prescribed minus actual rate, 2021–22 (percentage points)")
    ax.set_title("How far below the benchmarks was policy?")
    f3 = savefig(fig, "03_average_gap", SRC)

    # ---- Write-up -----------------------------------------------------------------------------------
    md.append("### Headline numbers\n")
    md.append(f"- Actual liftoff: **{liftoff:%B %Y}** ({liftoff.to_period('Q')})." if pd.notna(liftoff) else "- No liftoff in sample.")
    md.append(f"- Across benchmarks, policy was on average **{tab['Avg gap 2021–22 (pp)'].min():.1f}–{tab['Avg gap 2021–22 (pp)'].max():.1f} pp** "
              f"below the prescribed rate over 2021–22, and **{tab['Quarters late'].median():.0f} quarters** late (median across benchmarks).")
    md.append(f"- Estimated Fed reaction function 1987–2019: smoothing ρ = **{rho:.2f}**, long-run inflation response "
              f"**{lr['core']:.2f}** (Taylor principle requires > 1), output-gap response **{lr['gap']:.2f}**.")
    md.append(f"- Gradient-boosting Fed trained only on 1987–2019 behaviour (rolling-origin CV RMSE {cv_rmse:.2f} pp; "
              "it cannot extrapolate beyond rates seen in training, so treat it as a lower bound).\n")
    md.append("### Benchmarks\n")
    md.append(md_table(tab, "{:.2f}"))
    md.append("\n### Estimated reaction function (1987Q3–2019Q4, HAC SEs)\n")
    coef = pd.DataFrame({"Term": ols.params.index, "Coef": ols.params.values, "SE": ols.bse.values, "p": ols.pvalues.values})
    md.append(md_table(coef, "{:.3f}"))
    md.append("\n### Figures\n")
    for f, cap in [(f2, "Behind the curve since 2019"), (f3, "Average gap by benchmark"), (f1, "Rules vs actual since 1960s")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
