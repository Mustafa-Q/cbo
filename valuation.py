from copula import generate_correlated_defaults, score_to_default_rate
from typing import List
import pandas as pd
from collections import defaultdict
from loan import Loan, ValuationLoan
from copy import deepcopy
from waterfall import Waterfall
import numpy as np
import numpy_financial as npf

# ----- Expected loss helper (horizon-adjusted PD * LGD) -----
def _expected_loss_horizon(loans, lgd_assumption: float = 0.70, horizon_weeks: int = 8) -> float:
    """Compute expected loss over the BNPL horizon using annual PDs mapped to
    horizon PDs and a constant LGD assumption consistent with simulation.
    EL = sum(EAD * PD_horizon * LGD).
    """
    from copula import score_to_default_rate  # local import to avoid cycles
    years = float(horizon_weeks) / 52.0
    total_el = 0.0
    for l in loans:
        ead = float(getattr(l, "order_amount", 0.0))
        cs = getattr(l, "credit_score", None)
        try:
            pd_ann = float(score_to_default_rate(cs))
        except Exception:
            pd_ann = 0.0
        pd_ann = max(0.0, min(pd_ann, 1.0))
        pd_h = 1.0 - (1.0 - pd_ann) ** years
        total_el += ead * pd_h * float(lgd_assumption)
    return float(total_el)

def compute_expected_tranche_npvs(tranches, reserve, loans: List[ValuationLoan], discount_rate=0.05):

    weeks = [2, 4, 6, 8]
    expected_cashflows = defaultdict(float)

    for loan in loans:
        cf = loan.expected_cashflows()
        for week, amount in cf.items():
            expected_cashflows[week] += amount

    # Clone tranches so we don't mutate originals
    temp_tranches = deepcopy(tranches)
    temp_reserve = deepcopy(reserve)

    # Apply to a new Waterfall
    total_loan_pool = sum(l.order_amount for l in loans)
    temp_waterfall = Waterfall(temp_tranches, temp_reserve, total_loan_pool)

    tranche_payment_streams = {tranche.name: [] for tranche in temp_tranches}

    for i, week in enumerate(weeks):
        payment = expected_cashflows[week]
        tranche_paid = temp_waterfall.apply_payments(payment, week)

        for t in temp_tranches:
            tranche_payment_streams[t.name].append(tranche_paid.get(t.name, 0))

    # Discount expected payments
    tranche_npvs = {}
    for t in temp_tranches:
        stream = tranche_payment_streams[t.name]
        npv = sum([amt / ((1 + discount_rate) ** ((i+1)/26)) for i, amt in enumerate(stream)])
        tranche_npvs[t.name] = npv

    return tranche_npvs



def compute_single_run_investor_metrics(tranches):
    """
    Build per-tranche investor metrics with:
    - upfront-outlay IRR convention ([-principal] + inflows)
    - automatic annualization based on inferred cashflow frequency
    """
    results = []

    for tranche in tranches:
        principal = tranche.principal

        # Prefer tranche-level cashflows if present (works for both SPV & MC paths)
        cf_attr = getattr(tranche, "cashflows", None)
        if isinstance(cf_attr, (list, tuple)) and any(abs(x) > 1e-12 for x in cf_attr):
            full_cashflow = list(cf_attr)

            # Infer payment frequency (weeks) from nonzero gaps for annualization
            nonzero_idx = [i for i, a in enumerate(full_cashflow) if abs(a) > 1e-9]
            if len(nonzero_idx) >= 2:
                gaps = [j - i for i, j in zip(nonzero_idx[:-1], nonzero_idx[1:]) if j > i]
                period_weeks = min(gaps) if gaps else 2
            else:
                period_weeks = 2
            freq_per_year = 52.0 / float(period_weeks)

            # IRR as upfront outlay + inflows
            irr_cashflow = [-float(principal)] + full_cashflow if full_cashflow else [-float(principal)]
            annual_irr = None
            try:
                if any(c > 0 for c in irr_cashflow[1:]):
                    periodic_irr = npf.irr(irr_cashflow)
                    if periodic_irr is not None:
                        annual_irr = (1.0 + periodic_irr) ** float(freq_per_year) - 1.0
            except Exception:
                annual_irr = None

            total_received = float(sum(full_cashflow))
            cash_on_cash = (total_received / principal) if principal > 0 else 0.0
            loss_severity = max(0.0, (principal - total_received) / principal) if principal > 0 else 0.0

            results.append({
                "Tranche": tranche.name,
                "Principal": round(principal, 2),
                "Total Received": round(total_received, 2),
                "Annual IRR (%)": round(annual_irr * 100, 2) if annual_irr is not None else None,
                "Cash-on-Cash Return (%)": round(cash_on_cash * 100, 2),
                "Loss Severity (%)": round(loss_severity * 100, 2),
            })
            continue

        total_received = 0.0

        # Collect total payments per week across all units
        full_cashflow = []
        max_week = 0
        all_weeks = []

        for unit in tranche.units:
            for week, amount in unit.cashflow_history.items():
                # grow list to accommodate week index
                while len(full_cashflow) <= week:
                    full_cashflow.append(0.0)
                full_cashflow[week] += float(amount)
                max_week = max(max_week, week)
                all_weeks.append(int(week))

        # Trim trailing zeros to last week with activity
        full_cashflow = full_cashflow[:max_week + 1] if max_week >= 0 else []

        # ---- infer frequency for annualization ----
        # If we have explicit week indices, infer period length from gaps; else heuristic:
        if len(all_weeks) >= 2:
            sorted_w = sorted(set(all_weeks))
            gaps = [j - i for i, j in zip(sorted_w[:-1], sorted_w[1:]) if (j - i) > 0]
            period_weeks = min(gaps) if gaps else 2  # default biweekly if all same week
            freq_per_year = 52.0 / float(period_weeks)
        else:
            # Heuristic: if exactly four nonzero periods, likely 8-week/biweekly => 26 per year
            nonzero_periods = sum(1 for x in full_cashflow if abs(x) > 1e-9)
            if nonzero_periods == 4:
                freq_per_year = 26.0
            else:
                freq_per_year = 52.0  # weekly fallback

        # ---- IRR as true upfront outlay then inflows ----
        if len(full_cashflow) > 0:
            irr_cashflow = [-float(principal)] + list(full_cashflow)
        else:
            irr_cashflow = [-float(principal)]

        # Compute periodic IRR; annualize by compounding freq_per_year periods
        annual_irr = None
        try:
            if any(c > 0 for c in irr_cashflow[1:]):
                periodic_irr = npf.irr(irr_cashflow)
                if periodic_irr is not None:
                    annual_irr = (1.0 + periodic_irr) ** float(freq_per_year) - 1.0
        except Exception:
            annual_irr = None

        total_received = float(sum(full_cashflow))
        cash_on_cash = (total_received / principal) if principal > 0 else 0.0
        loss_severity = max(0.0, (principal - total_received) / principal) if principal > 0 else 0.0

        results.append({
            "Tranche": tranche.name,
            "Principal": round(principal, 2),
            "Total Received": round(total_received, 2),
            "Annual IRR (%)": round(annual_irr * 100, 2) if annual_irr is not None else None,
            "Cash-on-Cash Return (%)": round(cash_on_cash * 100, 2),
            "Loss Severity (%)": round(loss_severity * 100, 2),
        })

    return pd.DataFrame(results)


# -------------------------------------
# Tranche Unit NPV Calculation Function
# -------------------------------------
def compute_tranche_unit_npvs(tranches, discount_rates):
  npv_records = []

  for tranche in tranches:
    tranche_rate = discount_rates.get(tranche.name, 0.05)
    biweekly_rate = (1 + tranche_rate) ** (1/26) - 1

    for i, unit in enumerate(tranche.units):
      npv = 0
      for week, payment in unit.cashflow_history.items():
        biweekly_period = week // 2
        discount_factor = (1 + biweekly_rate) ** biweekly_period
        npv += payment / discount_factor

      npv_records.append({
          "tranche": tranche.name,
          "unit_id": f"{tranche.name} - {i+1}",
          "npv": round(npv, 2),
          "fully_paid": unit.paid,
          "remaining_principal": round(unit.remaining_principal, 2)
      })

  return pd.DataFrame(npv_records)


def calculate_npv_module(cashflows_df, loans, annual_discount_rates=None, expected_loss=None, realized_loss=None):
    """
    Calculates Net Present Value (NPV) for all loan cashflows combined.
    Includes initial loan disbursement as a negative cashflow.
    Robust to missing/empty cashflow columns by reconstructing from loans' cashflow_history.
    """
    if annual_discount_rates is None:
        annual_discount_rates = [0.05, 0.10, 0.15]

    # Fallbacks for losses if not supplied by caller
    if expected_loss is None:
        expected_loss = _expected_loss_horizon(loans, lgd_assumption=0.70, horizon_weeks=8)

    if realized_loss is None:
        total_principal_rl = float(sum(getattr(loan, "order_amount", 0.0) for loan in loans))
        total_received_rl = 0.0
        for loan in loans:
            pr = getattr(loan, "payment_record", {}) or {}
            if isinstance(pr, dict):
                total_received_rl += float(sum(pr.values()))
        realized_loss = max(0.0, total_principal_rl - total_received_rl)

    # Try to read a cashflow column, fallback to reconstruction from loans
    cash_col = None
    if isinstance(cashflows_df, pd.DataFrame):
        for candidate in ("cashflow", "payment", "cashflows", "amount"):
            if candidate in cashflows_df.columns:
                cash_col = candidate
                break

    use_df = False
    if cash_col is not None:
        arr = cashflows_df[cash_col].astype(float).values
        use_df = bool(arr.size) and (abs(arr).sum() > 1e-9)
    else:
        arr = np.array([], dtype=float)

    if not use_df:
        # Rebuild weekly flows from each loan's cashflow_history
        weekly = defaultdict(float)
        for loan in loans:
            history = getattr(loan, "cashflow_history", None)
            if history:
                for week, amt in history.items():
                    weekly[int(week)] += float(amt)
        if weekly:
            max_week = max(weekly.keys())
            arr = np.zeros(max_week + 1, dtype=float)
            for w, amt in weekly.items():
                arr[int(w)] = amt
        else:
            arr = np.array([0.0], dtype=float)

    # Insert initial outflow at t=0
    total_principal = -float(sum(getattr(loan, "order_amount", 0.0) for loan in loans))
    adjusted = np.insert(arr, 0, total_principal).astype(float)

    # Discount with weekly-compounded rates
    npv_results = {}
    for annual_rate in annual_discount_rates:
        weekly_rate = (1.0 + annual_rate) ** (1.0 / 52.0) - 1.0
        npv_results[f"{int(annual_rate * 100)}%"] = float(npf.npv(weekly_rate, adjusted))

    loss_comparison = None
    if expected_loss is not None and realized_loss is not None:
        loss_comparison = {
            "expected_loss": expected_loss,
            "realized_loss": realized_loss,
            "difference": expected_loss - realized_loss,
        }

    return {"NPV_Results": npv_results, "Loss_Comparison": loss_comparison}


# --------------------------
# Correlated Defaults Utility
# --------------------------
def assign_correlated_defaults(loans, rho=0.2, seed=None):
    """
    Assign correlated default outcomes to each loan using Gaussian copula.

    Parameters:
        loans (list): List of Loan or ValuationLoan objects
        rho (float): Correlation between loans
        seed (int or None): Random seed
    """
    n_loans = len(loans)
    default_probs = [score_to_default_rate(loan.credit_score) for loan in loans]
    default_flags = generate_correlated_defaults(n_loans, default_probs, rho=rho, seed=seed)

    for loan, defaulted in zip(loans, default_flags):
        if defaulted:
            loan.set_default()