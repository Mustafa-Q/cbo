from copula import generate_correlated_defaults, score_to_default_rate
from typing import List
import pandas as pd
from collections import defaultdict
from loan import Loan

def calculate_summary_statistics(loans: List[Loan]):
    stats = {}

    def _loan_total_paid(loan):
        pr = getattr(loan, "payment_record", None)
        if isinstance(pr, dict) and pr:
            return float(sum(pr.values()))
        # Fallback: compute from paid_weeks and installment/order_amount
        paid_weeks = list(getattr(loan, "paid_weeks", []))
        if hasattr(loan, "installment"):
            installment = float(getattr(loan, "installment", 0.0))
        else:
            # assume 4 installments if not explicitly present
            order_amt = float(getattr(loan, "order_amount", 0.0))
            installment = order_amt / 4.0 if order_amt else 0.0
        return installment * len(set(paid_weeks))

    total_principal = sum(float(getattr(loan, "order_amount", 0.0)) for loan in loans)
    total_received = sum(_loan_total_paid(loan) for loan in loans)
    stats['average_recovery_rate'] = total_received / total_principal if total_principal > 0 else 0

    # Default rate by credit score bucket
    buckets = {'<600': [], '600-699': [], '>=700': []}
    for loan in loans:
        cs = int(getattr(loan, "credit_score", 0))
        if cs < 600:
            buckets['<600'].append(loan)
        elif cs < 700:
            buckets['600-699'].append(loan)
        else:
            buckets['>=700'].append(loan)

    stats['default_rate_by_bucket'] = {
        bucket: (sum(1 for l in loans_list if getattr(l, "defaulted", False)) / len(loans_list) if loans_list else 0)
        for bucket, loans_list in buckets.items()
    }

    # Total interest/late fees collected
    stats['total_late_fees_collected'] = sum(float(getattr(loan, "total_late_fees_paid", getattr(loan, "late_fees_collected", 0.0))) for loan in loans)

    # Effective APR (approximate)
    aprs = []
    for loan in loans:
        total_paid = _loan_total_paid(loan)
        order_amt = float(getattr(loan, "order_amount", 0.0))
        if order_amt > 0 and total_paid > 0:
            apr = ((total_paid / order_amt) - 1.0) * (52.0 / 8.0)  # 8 weeks horizon approx
            aprs.append(apr)
    stats['average_effective_apr'] = (sum(aprs) / len(aprs)) if aprs else 0.0

    return stats


# --------------------------
# Tranche Reporting Function
# --------------------------
def generate_reports(loans, tranches, reserve, waterfall):
    # -------------------------------
    # Loan-Level Summary
    # -------------------------------
    loan_summaries = []
    for loan in loans:
        # Robust attribute access with sensible defaults
        defaulted = bool(getattr(loan, "defaulted", False))
        prepaid = bool(getattr(loan, "prepaid", False))
        remaining_balance = float(getattr(loan, "remaining_balance", 0.0))
        late_weeks = list(getattr(loan, "late_weeks", []))
        paid_weeks = list(getattr(loan, "paid_weeks", []))
        total_late_fees_paid = float(getattr(loan, "total_late_fees_paid", getattr(loan, "late_fees_collected", 0.0)))
        payment_record = dict(getattr(loan, "payment_record", {}))
        credit_score = getattr(loan, "credit_score", None)
        loan_id = getattr(loan, "loan_id", None)
        order_amount = float(getattr(loan, "order_amount", 0.0))

        # Derive status
        if defaulted:
            status = "defaulted"
        elif prepaid or remaining_balance <= 1e-8:
            status = "paid"
        else:
            status = "active"

        loan_summaries.append({
            "loan_id": loan_id,
            "order_amount": order_amount,
            "credit_score": credit_score,
            "status": status,
            "remaining_balance": round(remaining_balance, 2),
            "missed_payments": len(set(late_weeks)),
            "late_weeks": sorted(set(late_weeks)),
            "paid_weeks": sorted(set(paid_weeks)),
            "total_late_fees_paid": round(total_late_fees_paid, 2),
            "payment_record": payment_record
        })

    loan_df = pd.DataFrame(loan_summaries)

    # -------------------------------
    # Tranche-Level Summary
    # -------------------------------
    tranche_summaries = []
    for tranche in tranches:
        total_paid = sum(unit.face_value - unit.remaining_principal for unit in tranche.units)
        tranche_summaries.append({
            "tranche_name": tranche.name,
            "principal": tranche.principal,
            "remaining": round(sum(unit.remaining_principal for unit in tranche.units), 2),
            "total_paid": round(total_paid, 2),
            "percent_paid": round((total_paid / tranche.principal) * 100, 2)
        })
    tranche_df = pd.DataFrame(tranche_summaries)

    # -------------------------------
    # Reserve Account History
    # -------------------------------
    reserve_df = pd.DataFrame([
        {"week": k, "reserve_balance": v} for k, v in reserve.history.items()
    ])

    # -------------------------------
    # Weekly Waterfall History
    # -------------------------------
    waterfall_df = pd.DataFrame(waterfall.history)

    return loan_df, tranche_df, reserve_df, waterfall_df




# -----------------------------
# Securities Reporting Function
# -----------------------------0
def generate_security_report(tranches):
    records = []
    for tranche in tranches:
        for i, unit in enumerate(tranche.units):
            records.append({
                "tranche": tranche.name,
                "unit_id": f"{tranche.name}-{i+1}",
                "face_value": unit.face_value,
                "remaining_principal": unit.remaining_principal,
                "paid_off": unit.paid
            })
    return pd.DataFrame(records)


# -----------------------------------------------
# Assign Correlated Defaults using Gaussian Copula
# -----------------------------------------------
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