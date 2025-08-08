from copula import generate_correlated_defaults, score_to_default_rate
from typing import List
import pandas as pd
from collections import defaultdict
from loan import Loan, ValuationLoan
from copy import deepcopy
from waterfall import Waterfall
import numpy as np
import numpy_financial as npf

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
    results = []

    for tranche in tranches:
        principal = tranche.principal
        total_received = 0
        full_cashflow = []

        # Build a list of total payments per week across all units
        max_week = 0
        for unit in tranche.units:
            for week, amount in unit.cashflow_history.items():
                while len(full_cashflow) <= week:
                    full_cashflow.append(0)
                full_cashflow[week] += amount
                max_week = max(max_week, week)

        full_cashflow = full_cashflow[:max_week + 1]

        # Construct IRR-style cashflow: initial outlay negative, then inflows
        # Pad to a full year (optional but safer)
        if len(full_cashflow) < 52:
            full_cashflow += [0] * (52 - len(full_cashflow))

        npv_cashflow = [ -principal ] + full_cashflow

        # Only attempt IRR if there's at least one positive inflow
        if any(c > 0 for c in npv_cashflow[1:]):
            try:
                biweekly_irr = npf.irr(npv_cashflow)
                annual_irr = (1 + biweekly_irr) ** 26 - 1 if biweekly_irr is not None else None
            except:
                biweekly_irr = None
                annual_irr = None
        else:
            biweekly_irr = None
            annual_irr = None

        total_received = sum(full_cashflow)
        cash_on_cash = total_received / principal if principal > 0 else 0
        loss_severity = max(0, (principal - total_received) / principal)

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
    """
    if annual_discount_rates is None:
        annual_discount_rates = [0.05, 0.10, 0.15]

    weeks_per_year = 52
    weekly_rates = [(1 + rate)**(1/weeks_per_year) - 1 for rate in annual_discount_rates]

    total_cashflows = cashflows_df['cashflow'].values
    
    # Add initial outflow (principal disbursed)
    total_principal = -sum(loan.order_amount for loan in loans)
    adjusted_cashflows = np.insert(total_cashflows, 0, total_principal)

    npv_results = {}
    for rate, annual_rate in zip(weekly_rates, annual_discount_rates):
        npv_results[f"{int(annual_rate*100)}%"] = npf.npv(rate, adjusted_cashflows)

    loss_comparison = None
    if expected_loss is not None and realized_loss is not None:
        loss_comparison = {
            'expected_loss': expected_loss,
            'realized_loss': realized_loss,
            'difference': expected_loss - realized_loss
        }

    return {
        "NPV_Results": npv_results,
        "Loss_Comparison": loss_comparison
    }


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