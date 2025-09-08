# CBO — Collateralized BNPL Obligations

A compact simulation and valuation engine for a BNPL (Buy Now, Pay Later) loan pool with correlated defaults, behavioral payment logic, a tranche waterfall with a reserve account, and investor metrics (NPV/IRR). It can run a single SPV-style path or Monte Carlo scenarios to price tranches and analyze risk.

## Features

- BNPL loan modeling with biweekly payments (weeks 2/4/6/8), prepayment, late fees, missed-pay thresholds, and recoveries on default.
- Two loan flavors:
  - `Loan`: path-based behavior (late fees, misses, threshold default).
  - `ValuationLoan`: valuation-friendly with copula-scheduled default time and the same payment mechanics.
- Correlated defaults using Gaussian and t-copulas (`copula.py`), plus a hybrid default-time scheduler (`assign_correlated_default_times_hybrid`).
- Hazard/PD models including a logistic PD map and a simplified Cox proportional hazard (`hazard_models.py`).
- Tranche waterfall with Senior/Mezzanine/Equity and a `ReserveAccount` with top-up policy (`waterfall.py`, `tranche.py`).
- Valuation tools for per-tranche NPV, unit NPV, and investor metrics including annualized IRR (`valuation.py`).
- Portfolio statistics, bucketed default rates, and summaries (`statistics.py`).

## Project Structure

- `full_loan_simulation.py` — End-to-end demo: generate loans, assign correlated default times, simulate payments, run waterfall, metrics, and risk summary.
- `loan.py` — `Loan` and `ValuationLoan` classes (payment logic, default/prepay behavior, recovery on default).
- `copula.py` — Gaussian and t-copula correlated default generators.
- `hazard_models.py` — Score-to-PD mapping and simplified Cox hazard rate.
- `credit_state.py` — Simple credit state Markov transitions (optional downgrade behavior).
- `helpers.py` — Cashflow projection, aggregation, tranche waterfall simulator, summary printer, and basic distribution fitting.
- `valuation.py` — Tranche NPV, tranche unit NPV, investor metrics (IRR/CoC), and pool NPV utilities.
- `statistics.py` — Pool statistics, robust default detection, and report builders.
- `waterfall.py` — `ReserveAccount` and `Waterfall` with sequential pay and reserve top-up policy.
- `tranche.py` — Tranche and unit primitives used by the waterfall.

## Financial Background (Plain-English)

This project mirrors a simplified collateralized debt obligation (CDO) for short-dated, interest-free BNPL receivables. Here’s the intuition and how it maps to the code.

### Securitization and SPV

You take many small loans and put them into a bankruptcy-remote vehicle (an SPV). Investors buy notes issued by the SPV. The SPV uses borrower payments to pay investors. In code:

- Pool: a list of `Loan`/`ValuationLoan` objects with biweekly installments at weeks 2/4/6/8 (`loan.py`).
- SPV run: aggregate borrower cashflows, then pay investors via a waterfall (`full_loan_simulation.py`, `helpers.py`, `waterfall.py`).

### Tranching and Waterfall

A CDO splits investor claims into tranches with different seniority:

- Senior gets paid first, then Mezzanine, then Equity (the first-loss piece).
- Losses hit Equity first; Senior is protected unless losses are severe.
- A reserve account can be used to smooth shortfalls before paying Equity.

In code:

- `tranche.py`: simple tranche and unit primitives.
- `waterfall.py`: sequential pay waterfall; weekly interest to outstanding tranche balances, then principal; reserve top-up occurs before any equity residual is paid.
- `helpers.simulate_tranche_waterfall`: a lightweight waterfall used by both the SPV path and Monte Carlo valuation.

### BNPL Specifics: Interest‑Free, Short Maturity

Unlike credit cards or term loans, BNPL here is modeled as:

- Four interest‑free installments over ~8 weeks. Borrowers don’t accrue asset interest; they repay principal on due dates.
- Revenue and variability to the SPV arise from late fees and recoveries on default; not from contractual borrower interest.
- Investor “rates” in this repo (e.g., 5%, 8%, 15%) are not charged to borrowers; they’re discount/yield assumptions used to value tranche cashflows at the SPV level.

In code:

- Payment weeks fixed at 2/4/6/8; optional prepay on due dates.
- Late fees accrue when obligations are 2+ weeks overdue, capped at 25% of order amount (`loan.py`).
- Default triggers after repeated misses or at a copula‑scheduled default time; recovery is a fixed fraction (e.g., 30%).

### Default Risk: PD, Hazard, Correlation

- Probability of Default (PD): mapped from credit score by a logistic function (`hazard_models.score_to_default_rate`).
- Hazard model: simplified Cox hazard scales risk by credit score; we convert annualized hazard/PD into an 8‑week horizon PD used by the hybrid scheduler.
- Due‑date risk tilt: a `due_hazard_multiplier` increases hazard on payment weeks to reflect liquidity stress around due dates.
- Correlated defaults: a copula couples borrower outcomes via a shared factor. Gaussian copula captures correlation; t‑copula also captures tail dependence. Parameter `rho` controls the strength of co‑movement.

## Theory Deep Dive

### Copulas and Correlated Defaults

- Intuition: Each borrower has a latent normal (or t) variable capturing credit strength. Individuals default when that variable falls below a threshold consistent with their marginal PD. A copula ties these latent variables together so defaults co‑move.
- Gaussian Copula: Build a correlation matrix using `rho`; correlate standard normals by Cholesky decomposition. A borrower i defaults if `Z_corr[i] < Φ⁻¹(PD_i)`, preserving marginal PDs and embedding linear correlation. See `copula.generate_correlated_defaults`.
- t‑Copula (Tail Dependence): Replace normals with Student‑t draws. Mapping to uniforms via the t CDF introduces tail dependence: extreme joint events (many defaults) become more likely than under Gaussian. See `copula.generate_t_copula_defaults`.
- Practical impact: Higher `rho` or lower t degrees‑of‑freedom increases clustered defaults, which disproportionately threaten Senior tranches.

### Hazard Rates, PD, and Default Time

- PD mapping: `hazard_models.score_to_default_rate` converts credit score to annual PD via a logistic curve (bounded away from 0/1 for stability).
- Cox‑style hazard: `hazard_models.cox_hazard_rate` scales a baseline hazard `λ₀` by `exp(β·x)`, where `x` is the normalized score. This provides relative risk across borrowers.
- Horizon conversion: `full_loan_simulation.annual_to_horizon_pd` maps annual PD to an 8‑week PD. `weekly_hazard_from_horizon_pd` solves for weekly hazard p such that `1 − (1 − p)^W = PD_horizon`.
- Due‑date tilt: We split weekly hazard into `p_due` on weeks {2,4,6,8} and `p_other` elsewhere using `due_hazard_multiplier`, concentrating risk around payment dates.
- Inverse‑transform time selection: Draw U~Uniform(0,1); with weekly hazards `{p_w}`, the default week is the first w where cumulative default probability exceeds U. We then snap cash accounting to the next due week for realism.

### Loss, Recovery, and Expected Loss (EL)

- Recovery: On default we realize a fixed recovery (e.g., 30%) immediately; remaining principal is written off.
- LGD: Loss‑Given‑Default = 1 − recovery (≈ 70%).
- EL alignment: `valuation._expected_loss_horizon` computes EL = sum(EAD × PD_horizon × LGD). `valuation.calculate_npv_module` compares EL to realized losses from simulated cashflows.

### Waterfall Mechanics and Reserve Policy

- Sequencing: For each cashflow date, pay interest on outstanding Senior/Mezzanine, then their principal; top‑up the reserve to target; pay any residual to Equity.
- Reserve trade‑off: A higher reserve target reduces timing shortfalls hitting Senior but diverts cash from Equity.

### Model Boundaries

- This is didactic. Hazard/PD mappings, recovery, and fee rules are simplified; real‑world BNPL may include merchant fees, interchange, servicing costs, and dynamic credit line management.

## Mathematical Appendix

### Copula Construction (Gaussian and t)

- Sklar’s theorem: Any multivariate distribution with marginals `F_i` can be written as `F(x_1,…,x_n) = C(F_1(x_1),…,F_n(x_n))` for some copula `C` on `[0,1]^n`.
- Gaussian copula with correlation matrix `Σ`: `C_Σ(u) = Φ_Σ(Φ^{-1}(u_1),…,Φ^{-1}(u_n))`, where `Φ_Σ` is the MVN CDF and `Φ` is univariate standard normal CDF. Simulation:
  - Build equicorrelation `Σ = ρ·11ᵀ + (1−ρ)·I` and Cholesky `L`.
  - Draw `Z_indep ~ N(0,I)`; set `Z_corr = L Z_indep`; set `U_i = Φ(Z_corr[i])`.
  - Default if `Z_corr[i] < θ_i` with `θ_i = Φ^{-1}(PD_i)` to preserve marginal `PD_i`.
- t‑copula with correlation `Σ` and `ν` dof: draw `T ~ t_ν(0, Σ)`, set `U_i = t_ν(T_i)`. Bivariate upper tail dependence coefficient is `λ_U = 2·t_{ν+1}(-√((ν+1)(1−ρ)/(1+ρ)))`, showing non‑zero tail dependence for finite `ν`.

LaTeX:

$$
\text{Sklar:}\quad F(x_1,\dots,x_n)=C\big(F_1(x_1),\dots,F_n(x_n)\big)
$$

$$
\text{Gaussian copula:}\quad C_\Sigma(\mathbf u)=\Phi_\Sigma\big(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_n)\big)
$$

$$
\Sigma=\rho\,\mathbf 1\mathbf 1^\top+(1-\rho)\,I,\qquad Z_{\text{corr}}=L\,Z_{\text{indep}},\quad \text{default if } Z_{\text{corr},i}<\Phi^{-1}(\text{PD}_i)
$$

$$
\text{t-copula tail dependence:}\quad \lambda_U=2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)
$$

### PD, Hazard, Survival, and Conversions

- Continuous time: hazard `h(t)`, survival `S(t) = exp(-∫₀^t h(s) ds)`, default CDF `F(t) = 1 − S(t)`.
- Discrete weeks: weekly hazard `p_w = P(default at w | survive to w−1)`. Over `W` weeks, `PD_W = 1 − ∏_{w=1}^W (1 − p_w)`.
- Constant weekly hazard `p`: `PD_W = 1 − (1 − p)^W` ⇒ `p = 1 − (1 − PD_W)^{1/W}`.
- Annual to horizon (8 weeks): given annual PD `PD_ann`, horizon years `T = 8/52`, `PD_8w = 1 − (1 − PD_ann)^T`.
- Weekly hazard from horizon PD: `p_week = 1 − (1 − PD_8w)^{1/8}`.
- Cox proportional hazards: `h_i = h₀ · exp(β x_i)` where `x_i` is normalized score. Implementation returns a bounded discrete proxy used to derive `PD_8w` via conversions above.

LaTeX:

$$
S(t)=\exp\Big(-\int_0^t h(s)\,ds\Big),\qquad F(t)=1-S(t)
$$

$$
\text{Discrete weeks:}\quad \text{PD}_W=1-\prod_{w=1}^{W}(1-p_w)
$$

$$
\text{Constant }p:\quad \text{PD}_W=1-(1-p)^W\;\Rightarrow\;p=1-(1-\text{PD}_W)^{1/W}
$$

$$
\text{Annual}\to\text{horizon (}T=8/52\text{):}\quad \text{PD}_{8w}=1-(1-\text{PD}_{\text{ann}})^T,
\qquad p_{\text{week}}=1-(1-\text{PD}_{8w})^{1/8}
$$

### Hybrid Default‑Time Scheduler

- Step 1 (systematic risk): Draw a common factor and idiosyncratic shocks; form `z_i = √ρ·M + √(1−ρ)·ε_i` and uniforms `U_i = Φ(z_i)`.
- Step 2 (per‑loan hazards): Compute `p_due` on due weeks `{2,4,6,8}` and `p_other` elsewhere from a base weekly hazard and `due_hazard_multiplier`.
- Step 3 (inverse transform): Find the smallest week `w` with `1 − ∏_{k=1}^w (1 − p_k) ≥ U_i`. Place default uniformly within week for a continuous timestamp; for cash, snap to the next due week.

LaTeX:

$$
z_i=\sqrt{\rho}\,M+\sqrt{1-\rho}\,\varepsilon_i,\quad U_i=\Phi(z_i),\quad
\tau_i=\min\Big\{w:\;1-\prod_{k=1}^{w}(1-p_k)\ge U_i\Big\}
$$

### Payment Dynamics and Fees

- Due stack: On each due week add `(due_week, installment, accrued_late_fee)` to the stack. Each week, if an item is ≥ 2 weeks late, increment its late fee by `min(late_fee, 0.25·order − total_late_fees_paid)` to respect the 25% cap.
- Miss events: With credit‑score‑bucketed miss probability, either pay full obligation or roll it forward. Count misses; when a threshold is reached (bucket‑dependent), trigger default.
- Prepayment: On each due date, with small probability (default 5%), prepay all remaining balance.
- Recovery: On default, immediately realize `recovery_rate · remaining_balance` as cash; set remaining balance to zero.

### Waterfall Math

- Weekly interest accrual per tranche on outstanding balance `B_t`: `I_t = B_t · (r_annual / 52)`. Pay interest first, then principal from available cash.
- Reserve policy: After paying Senior/Mezz interest and principal, top up reserve to target `R*` from remaining cash; any residual goes to Equity.
- Equity as residual: `Equity_t = max(0, cash_after_reserve)`.

LaTeX:

$$
I_t=B_t\cdot\frac{r_{\text{ann}}}{52},\qquad \text{NPV}=\sum_{t=0}^{T}\frac{\text{CF}_t}{(1+r_w)^t},\quad r_w=(1+r_{\text{ann}})^{1/52}-1
$$

### Valuation Math (NPV, IRR)

- Discounting: Convert annual discount to weekly `r_w = (1 + r_ann)^{1/52} − 1`. NPV is `Σ_{t=0}^T CF_t / (1 + r_w)^t`, with `CF_0` including the negative principal outlay.
- IRR: Solve `Σ_{t=0}^T CF_t / (1 + r)^t = 0` for periodic `r`. Annualize as `(1 + r)^{freq_per_year} − 1` where `freq_per_year ≈ 52 / period_weeks` inferred from observed payment gaps.

LaTeX:

$$
\sum_{t=0}^{T}\frac{\text{CF}_t}{(1+r)^t}=0,\qquad r_{\text{annual}}=(1+r)^{\text{freq/yr}}-1
$$

### Expected Loss and Realized Loss

- Per‑loan EL: `EL_i = EAD_i · PD_8w · LGD` with `LGD = 1 − recovery_rate`.
- Pool EL: sum over loans; compare to realized loss `max(0, Σ EAD_i − Σ collected_cash)` to assess calibration consistency.

LaTeX:

$$
\text{EL}_i=\text{EAD}_i\cdot \text{PD}_{8w}\cdot \text{LGD},\qquad \text{EL}_{\text{pool}}=\sum_i \text{EL}_i
$$

$$
\text{Realized Loss}=\max\!\left(0,\sum_i \text{EAD}_i-\sum_t \text{CF}_t\right)
$$

## LaTeX–to–Code Mapping

- Copulas (Gaussian/t): `copula.py` — functions `generate_correlated_defaults`, `generate_t_copula_defaults` implement the Cholesky correlation and thresholding with `norm.ppf` / `student_t.cdf`.
- Hazard and PD conversions: `hazard_models.py` — `score_to_default_rate`, `cox_hazard_rate`; `full_loan_simulation.py` — `annual_to_horizon_pd`, `weekly_hazard_from_horizon_pd`.
- Hybrid scheduler: `full_loan_simulation.py` — `assign_correlated_default_times_hybrid` implements `U_i=Φ(z_i)`, week‑wise inverse transform, due‑week tilt, and snapping.
- Payment dynamics and recovery: `loan.py` — late‑fee accrual with cap, missed‑payment thresholds, prepayment, and immediate recovery.
- Waterfall and reserve: `helpers.py` — `simulate_tranche_waterfall`; `waterfall.py` — `Waterfall.apply_payments` (interest → principal → reserve top‑up → equity residual).
- Valuation and EL: `valuation.py` — `calculate_npv_module`, `_expected_loss_horizon`, and `compute_single_run_investor_metrics` (NPV, IRR annualization).

### Monte Carlo Estimation

- Tranche price estimator: `E[NPV] ≈ (1/N) Σ_{n=1}^N NPV^{(n)}`. Sampling error decays as `O(1/√N)`; use larger `n_paths` for tighter confidence.

## Quick Start

Run the end-to-end demo and print summaries, tranche metrics, and risk stats:

```bash
python full_loan_simulation.py
```

You’ll see output similar to:
- Simulation summary over multiple runs
- Copula impact summary (defaults, tail loss, equity outcomes)
- Risk metrics (P50, P95, ES95)
- Tranche investor metrics (annual IRR, cash-on-cash, loss severity)

## Minimal Programmatic Example

```python
import pandas as pd
from loan import ValuationLoan
from full_loan_simulation import assign_correlated_default_times_hybrid
from helpers import simulate_tranche_waterfall
from full_loan_simulation import aggregate_cashflows_local

# 1) Build a tiny portfolio
loans = [
    ValuationLoan(loan_id=1, order_amount=400.0, credit_score=650),
    ValuationLoan(loan_id=2, order_amount=600.0, credit_score=720),
]

# 2) Correlated default-time assignment (hybrid copula + hazard)
assign_correlated_default_times_hybrid(loans, rho=0.3, seed=42, due_hazard_multiplier=1.5)

# 3) Simulate payments on due weeks
for week in (2, 4, 6, 8):
    for l in loans:
        l.simulate_payment(week)

# 4) Aggregate pool cashflows
cashflow_df = aggregate_cashflows_local(loans)  # columns: week, cashflow

# 5) Run a simple tranche waterfall
total_pool = sum(l.order_amount for l in loans)
tranche_structure = [
    {"name": "Senior",    "principal": 0.50 * total_pool, "rate": 0.06},
    {"name": "Mezzanine", "principal": 0.30 * total_pool, "rate": 0.10},
    {"name": "Equity",    "principal": 0.20 * total_pool, "rate": 0.15},
]
tranche_cashflows = simulate_tranche_waterfall(cashflow_df, tranche_structure)
print(pd.DataFrame(tranche_cashflows))
```

## Key Concepts and Knobs

- Correlation: `rho` (e.g., 0.05–0.30 typical). Higher = more tail risk.
- Due-week hazard tilt: `due_hazard_multiplier` in `assign_correlated_default_times_hybrid` tilts default probabilities toward due weeks.
- Recovery: On default, a simple recovery (e.g., 30%) is realized immediately.
- Late fees: Accrue once an installment is 2+ weeks overdue, capped at 25% of order amount.
- Prepayment: Small chance at scheduled payment dates (default 5%).
- Tranche rates: Weekly interest applied to outstanding balances before principal.
- Reserve policy: Top-up to target before equity residual is paid.
- LGD for expected loss: Assumed 70% in valuation helpers for pool NPV.

These assumptions live primarily in `loan.py`, `full_loan_simulation.py`, `helpers.py`, and `valuation.py` and can be adjusted for your use case.

## Monte Carlo Valuation

Use `monte_carlo_tranche_npvs` in `full_loan_simulation.py` to price tranches via repeated paths that share the same mechanics as the single-run SPV logic:

```python
from full_loan_simulation import simulate_loans, monte_carlo_tranche_npvs
import pandas as pd

loans_df = simulate_loans(num_loans=200)
tranche_structure = [
    {"name": "Senior",    "principal": 0.40 * loans_df['order_amount'].sum(), "rate": 0.05},
    {"name": "Mezzanine", "principal": 0.30 * loans_df['order_amount'].sum(), "rate": 0.08},
    {"name": "Equity",    "principal": 0.30 * loans_df['order_amount'].sum(), "rate": 0.15},
]
npvs = monte_carlo_tranche_npvs(loans_df, tranche_structure, rho=0.07, n_paths=200)
print(npvs)
```

## Reports and Metrics

- Tranche investor metrics: `valuation.compute_single_run_investor_metrics(tranches)`
- Tranche unit NPVs: `valuation.compute_tranche_unit_npvs(tranches, discount_rates)`
- Pool NPV and loss comparison: `valuation.calculate_npv_module(cashflows_df, loans, ...)`
- Portfolio statistics (defaults by score bucket, recovery, APR): `statistics.calculate_summary_statistics(loans)`

## Data and Calibration

Helpers are included to fit simple empirical distributions you can plug in:
- `fit_income_distribution` (lognormal)
- `fit_default_time_distribution` (exponential)
- `fit_delay_distribution` (Poisson)

You can replace the synthetic draws in `simulate_loans` with your own empirical samples.

## Notes and Limitations

- This is a pedagogical, compact model; it is not production risk software. Many parameters (PD mapping, hazard, recovery, late fee rules) are simplified.
- The code intentionally favors clarity and single-file utilities; feel free to refactor into packages or add tests as needed.

## License

No license specified. If you plan to distribute or use commercially, add an appropriate license file.
