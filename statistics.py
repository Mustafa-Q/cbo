from copula import generate_correlated_defaults
from hazard_models import score_to_default_rate
from typing import List
import pandas as pd
from collections import defaultdict
from loan import Loan


def _is_defaulted(loan) -> bool:
    """
    Robust default detector for both Loan and ValuationLoan.
    Priority:
      1) Explicit flags set during sim
      2) Copula scheduling markers
      3) Fallback: unpaid balance after horizon (not prepaid)
    """
    # 1) Explicit flags
    if bool(getattr(loan, "defaulted", False)):
        return True
    if bool(getattr(loan, "externally_defaulted", False)):
        return True

    # 2) Copula / scheduled default markers (ValuationLoan)
    if getattr(loan, "default_week", None) is not None:
        return True
    if getattr(loan, "default_scheduled_week_to_pay", None) is not None:
        return True
    dtc = getattr(loan, "default_time_week_continuous", float("inf"))
    try:
        if float(dtc) != float("inf"):
            return True
    except Exception:
        pass

    # No fallback inference: only explicit flags or scheduled defaults count
    return False

from typing import Dict

def calculate_summary_statistics(loans: List[Loan]) -> Dict[str, object]:
    """
    Compatibility wrapper used by full_loan_simulation.py and others.
    Produces a dict with the same keys as the legacy implementation, but
    uses the new, robust `_is_defaulted` detector defined at module scope.
    """
    stats: Dict[str, object] = {}

    def _loan_total_paid(loan) -> float:
        pr = getattr(loan, "payment_record", None)
        if isinstance(pr, dict) and pr:
            return float(sum(pr.values()))
        # Fallback: compute from paid_weeks and installment/order_amount
        paid_weeks = list(getattr(loan, "paid_weeks", []))
        if hasattr(loan, "installment"):
            installment = float(getattr(loan, "installment", 0.0))
        else:
            order_amt = float(getattr(loan, "order_amount", 0.0))
            installment = order_amt / 4.0 if order_amt else 0.0
        return float(installment) * len(set(paid_weeks))

    # Pool-level cash summary
    total_principal = float(sum(float(getattr(l, "order_amount", 0.0)) for l in loans))
    total_received = float(sum(_loan_total_paid(l) for l in loans))

    stats['average_recovery_rate'] = (total_received / total_principal) if total_principal > 1e-12 else 0.0
    stats['realized_loss'] = max(0.0, total_principal - total_received)

    # Default rate by credit score bucket (robust)
    def _bucket_for_score(raw_cs):
        try:
            cs = int(raw_cs)
        except Exception:
            return ">=700" if raw_cs is None else "<600"  # fallback
        if cs < 600:
            return "<600"
        elif cs < 700:
            return "600-699"
        else:
            return ">=700"

    buckets = {'<600': [], '600-699': [], '>=700': []}
    for loan in loans:
        bucket = _bucket_for_score(getattr(loan, "credit_score", None))
        buckets[bucket].append(loan)

    defaults_by_bucket = {}
    count_by_bucket = {}
    rates_by_bucket = {}
    for bucket, loans_list in buckets.items():
        n = len(loans_list)
        count_by_bucket[bucket] = n
        d = sum(1 for l in loans_list if _is_defaulted(l))
        defaults_by_bucket[bucket] = d
        rates_by_bucket[bucket] = (d / n) if n > 0 else 0.0

    stats['defaults_by_bucket'] = defaults_by_bucket
    stats['count_by_bucket'] = count_by_bucket
    stats['default_rate_by_bucket'] = rates_by_bucket

    # Total late fees collected (support both attributes)
    stats['total_late_fees_collected'] = float(sum(
        float(getattr(l, "total_late_fees_paid", getattr(l, "late_fees_collected", 0.0)))
        for l in loans
    ))

    # Approximate average effective APR over the 8-week BNPL horizon
    aprs = []
    for l in loans:
        total_paid = _loan_total_paid(l)
        order_amt = float(getattr(l, "order_amount", 0.0))
        if order_amt > 1e-12 and total_paid > 1e-12:
            apr = ((total_paid / order_amt) - 1.0) * (52.0 / 8.0)  # weekly to annual (~8-week horizon)
            aprs.append(apr)
    stats['average_effective_apr'] = (sum(aprs) / len(aprs)) if aprs else 0.0

    return stats


# -----------------------------
# Compatibility report builders
# -----------------------------
import pandas as pd
from typing import Tuple

def generate_reports(loans, tranches, reserve, waterfall) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compatibility helper expected by full_loan_simulation.py.
    Returns: (loan_df, tranche_df, reserve_df, waterfall_df)
    Only the columns used downstream are guaranteed.
    """
    # Loan report (keep columns used by full_loan_simulation)
    loan_rows = []
    for l in loans:
        loan_rows.append({
            'loan_id': getattr(l, 'loan_id', None),
            'order_amount': float(getattr(l, 'order_amount', 0.0)),
            'credit_score': getattr(l, 'credit_score', None),
        })
    loan_df = pd.DataFrame(loan_rows)

    # Tranche report
    tranche_rows = []
    for t in tranches:
        cf = getattr(t, 'cashflows', None) or []
        total_received = float(sum(cf)) if cf else float(getattr(t, 'total_received', 0.0))
        tranche_rows.append({
            'name': getattr(t, 'name', 'Tranche'),
            'principal': float(getattr(t, 'principal', 0.0)),
            'total_received': total_received,
        })
    tranche_df = pd.DataFrame(tranche_rows)

    # Reserve report (very lightweight)
    res_bal = float(getattr(reserve, 'balance', 0.0)) if reserve is not None else 0.0
    reserve_df = pd.DataFrame([{'balance': res_bal}])

    # Waterfall report: prefer waterfall.history if available; else, aggregate from loans
    wf_rows = []
    history = getattr(waterfall, 'history', None)
    if isinstance(history, dict) and history:
        weeks = max((len(v) for v in history.values()), default=0)
        for w in range(weeks):
            total = 0.0
            for v in history.values():
                if w < len(v):
                    total += float(v[w])
            wf_rows.append({'week': w, 'total': total})
    else:
        from collections import defaultdict
        totals = defaultdict(float)
        for l in loans:
            pr = getattr(l, 'payment_record', {}) or {}
            if isinstance(pr, dict):
                for w, amt in pr.items():
                    try:
                        w_int = int(w)
                    except Exception:
                        continue
                    totals[w_int] += float(amt)
        for w in sorted(totals.keys()):
            wf_rows.append({'week': int(w), 'total': float(totals[w])})

    waterfall_df = pd.DataFrame(wf_rows)
    return loan_df, tranche_df, reserve_df, waterfall_df


def generate_security_report(*args, **kwargs):
    """
    Placeholder for backward compatibility. Not used by full_loan_simulation.py
    in your current workflow. Returns an empty DataFrame.
    """
    return pd.DataFrame(columns=['security', 'metric', 'value'])
