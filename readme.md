# CBO — Collateralized BNPL Obligations

This project simulates and values a short‑dated BNPL securitization with three tranches and a weekly waterfall.

## Overview

- Short‑maturity, non‑revolving deal backed by BNPL receivables.
- Receivables sit in an SPV that issues Senior, Mezzanine, and Equity notes.
- Borrowers pay no interest; SPV revenues come from late fees and recoveries. Investor rates here are valuation discount rates, not borrower coupons.

## Workflow

- Origination: purchases split into four installments due at weeks 2/4/6/8.
- Borrower actions: pay, prepay, pay late (capped fees), or default with partial recovery.
- Collection: payments aggregated weekly.
- Waterfall: weekly priority of interest, principal, reserve top‑up, then residual to Equity.
- No reinvestment; pool amortizes over ~8 weeks.

## Features

- Simulates borrower cashflows including prepay, late fees, defaults, and recoveries (with optional correlation).
- Routes cash through a sequential waterfall with a reserve policy.
- Values tranche cashflows and reports NPV/IRR for single runs and Monte Carlo.

## Use Cases

- Price tranches under different structures and assumptions.
- Run Monte Carlo to study sensitivity to defaults, recoveries, correlation, and reserve targets.
- Explore trade‑offs among tranche sizing, discount rates, and equity residual.

## Scope

- Short‑dated BNPL; no revolving period, hedging, or complex triggers.
- Minimal fees beyond what’s needed for cash routing.
- Borrower interest not modeled; revenues are late fees and recoveries.
- Stylized but tunable credit risk and correlation.

## Inputs

- Pool: loan count, order amounts, borrower scores.
- Credit: default tendencies, recovery rates, correlation.
- Structure: tranche sizes, valuation rates, reserve target.

## Outputs

- Weekly cashflows to each tranche and Equity.
- Per‑tranche NPV and IRR for single run and Monte Carlo averages.
- Portfolio summaries: defaults, losses, late‑fee totals, and distributional views.

## Quick Start

- Run: `python full_loan_simulation.py`
- Builds a sample pool, simulates payments, runs the waterfall, and prints tranche metrics and summaries.

## Files

- `full_loan_simulation.py`: Demo driver and scenario runner.
- `loan.py`: BNPL payment behaviour (schedule, prepay, late fees, default/recovery).
- `tranche.py`, `waterfall.py`: Tranche definitions and sequential waterfall with reserve.
- `valuation.py`: Tranche valuation and investor metrics.
- `copula.py`: Default correlation for stress/risk scenarios.
- `hazard_models.py`: Maps scores to default tendencies.
- `helpers.py`, `statistics.py`: Cashflow aggregation and reporting.
