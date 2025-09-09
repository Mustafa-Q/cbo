# CBO — Collateralized BNPL Obligations

This project simulates and values a short‑dated BNPL securitization. Think of it as a compact, readable lab for structuring and analyzing a pool of Buy Now, Pay Later installments that are packaged into notes (tranches) and paid by a simple waterfall.

## What The Security Is

- A short‑maturity, non‑revolving securitization backed by BNPL receivables.
- The receivables sit in a special‑purpose vehicle (SPV) that issues three notes:
  - Senior: paid first, most protected from losses.
  - Mezzanine: paid after Senior, some protection.
  - Equity: first‑loss piece; gets whatever is left over.
- BNPL is modeled as interest‑free to the borrower. Variability to the SPV comes from late fees and recoveries on default, not borrower interest. Investor “rates” in this repo are discount/valuation rates for pricing tranche cashflows, not borrower coupons.

## How It Works (End‑to‑End)

- Origination and Pool: Many small purchases are split into four equal installments due at weeks 2/4/6/8.
- Borrower Behavior: On each due week, a borrower can pay, prepay remaining balance, pay late and incur capped late fees, or default; on default, a fraction of the remaining balance is recovered.
- Collection: The SPV aggregates all borrower payments by week.
- Waterfall: Each week, cash is used to pay tranche interest and principal in order of priority; then the reserve account is topped up to its target; any residual goes to Equity.
- Lifecycle: There is no reinvestment; the deal winds down as loans amortize over the ~8‑week horizon.

## What This Code Is Trying To Do

- Generate realistic borrower cashflows for a BNPL pool, including prepayments, late fees, missed payments, recoveries, and defaults (with optional correlation across borrowers).
- Route the collected cash through a simple tranche waterfall with a reserve policy.
- Produce investor results for each tranche (cashflows, NPVs, IRRs) both for a single run and across many Monte Carlo scenarios for risk analysis.

## What You Can Use This For

- Price tranches and compare investor outcomes under different structures and assumptions.
- Run Monte Carlo to see the impact of default levels, recovery rates, correlation, or reserve targets on each tranche.
- Explore trade‑offs between tranche sizing, discount rates used for valuation, and equity residual.

## Scope and Simplifications

- Focused on short‑dated BNPL; no revolving period, hedging, or complex performance triggers/covenants.
- No tax, legal, accounting, or detailed servicing fee schedules beyond the essentials required for cash routing.
- Borrower interest is not modeled; revenues arise from late fees and recoveries.
- Credit risk and correlation are stylized but tunable; use this as a sandbox for structure design and intuition, not production risk management.

## Inputs You Control (High‑Level)

- Pool: number of loans, order amounts, and borrower credit scores.
- Credit: default tendencies, recovery rates, and how correlated borrower outcomes are.
- Structure: tranche sizes, target rates used for valuation, and reserve account target.

## Outputs You Get (High‑Level)

- Weekly cashflows to each tranche and to Equity.
- Investor metrics per tranche, such as NPV and IRR, for a single run and averaged across Monte Carlo paths.
- Portfolio summaries: defaults, losses, late‑fee totals, and basic distributional/risk views.

## Quick Start

- Run the end‑to‑end demo: `python full_loan_simulation.py`
- It builds a sample pool, simulates borrower payments, runs the waterfall, prints tranche NPVs and IRRs, and shows summary statistics.

## Files At A Glance

- `full_loan_simulation.py`: One‑button demo and scenario runner.
- `loan.py`: BNPL payment behavior (due schedule, prepay, late fees, default and recovery).
- `tranche.py` and `waterfall.py`: Tranche definitions and a sequential‑pay waterfall with a reserve account.
- `valuation.py`: Per‑tranche cashflow valuation and investor metrics.
- `copula.py`: Makes borrower defaults move together (default correlation) for stress/risk scenarios.
- `hazard_models.py`: Simple credit‑risk knobs that map scores to default tendencies.
- `helpers.py` and `statistics.py`: Cashflow aggregation and reporting utilities.

That’s it: a compact, readable securitization playground for BNPL where you can change pool, credit, and structure assumptions and immediately see how investor cashflows and values move.

