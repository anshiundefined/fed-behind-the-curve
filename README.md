# How Late Was the Fed? Measuring "Behind the Curve" in 2021–22

US core PCE inflation passed 3% in spring 2021 and peaked above 5%, yet the Federal Reserve kept rates at zero until **March 2022**. "The Fed was behind the curve" is now conventional wisdom. **Behind which curve, and by how much?**

This repo turns the phrase into numbers. It compares the actual fed funds rate against five benchmarks for where it "should" have been, including an ML model of how the Fed itself behaved before 2020.

## Data (FRED, pulled live)
| Series | Code |
|---|---|
| Effective fed funds rate | `FEDFUNDS` |
| Core PCE price index / headline PCE | `PCEPILFE` / `PCEPI` |
| Real GDP / CBO potential GDP | `GDPC1` / `GDPPOT` |
| Unemployment rate / CBO natural rate | `UNRATE` / `NROU` |

Quarterly, 1960s to latest. Inflation is year-on-year. The output gap is `100 × (GDP / potential − 1)`.

## Benchmarks
| Benchmark | Rule |
|---|---|
| Taylor (1993) | `i = 2 + π + 0.5(π − 2) + 0.5·gap` |
| Balanced approach (1999) | Same, with 1.0 on the output gap |
| Inertial Taylor | `i = 0.85·i₋₁ + 0.15·Taylor`, iterated on its own path |
| **Estimated Fed rule** | Partial-adjustment reaction function `i = c + ρ·i₋₁ + β·π + γ·gap` estimated on 1987–2019 (HAC SEs), then **simulated dynamically** from 2020 on its own lagged rate |
| **ML Fed** | Gradient boosting trained on 1987–2019 (core and headline inflation, output gap, unemployment gap, inflation momentum). It asks: *what would the Greenspan-to-Powell Fed have done with 2021–22 data?* |

All prescriptions are floored at zero, the effective lower bound. "Quarters late" is the gap between the first quarter each benchmark calls for rates of at least 0.75% and the actual liftoff.

## Results
<!-- RESULTS:START -->
_Results, tables and figures are generated automatically by the GitHub Action (`Actions` tab → **Run analysis**). They appear here a few minutes after the first push._
<!-- RESULTS:END -->

## Reproduce
```bash
pip install -r requirements.txt
python run.py
```
The GitHub Action re-pulls FRED and re-runs on push and monthly.

## Limitations
- Potential GDP and r* are unobserved and revised over time. This uses today's vintage (CBO potential, r* = 2%), not what the Fed saw in real time, which flatters hindsight. Orphanides (2003) shows how much real-time data matter.
- The 2020 framework change (flexible average inflation targeting) deliberately allowed overshooting, so a pre-2020 benchmark partly measures the *regime change* rather than a mistake.
- Tree models cannot extrapolate beyond rates seen in training, so the ML Fed is a conservative benchmark.
- Balance-sheet policy (QE/QT) is ignored. Shadow-rate estimates would change the pre-2022 picture.

## References
- Taylor, J. (1993). *Discretion versus policy rules in practice.* Carnegie-Rochester Conference Series.
- Clarida, R., Galí, J. & Gertler, M. (2000). *Monetary policy rules and macroeconomic stability.* QJE.
- Orphanides, A. (2003). *Historical monetary policy analysis and the Taylor rule.* Journal of Monetary Economics.
- Bernanke, B. & Blanchard, O. (2023). *What caused the US pandemic-era inflation?* Brookings/PIIE.
