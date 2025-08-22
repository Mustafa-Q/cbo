from typing import List
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import norm
from copula import generate_correlated_defaults
from hazard_models import score_to_default_rate


def project_loan_cashflows(loans, weeks, prepay_rate=0.01):
    expected_cashflows = {week: 0.0 for week in weeks}
    for loan in loans:  # loans is a list of ValuationLoan objects
        prob_default = score_to_default_rate(loan.credit_score)
        survival_prob = 1.0
        for i, week in enumerate(loan.expected_payment_schedule):
            # Skip weeks we aren't tracking
            if week not in expected_cashflows:
                continue
            expected_payment = loan.installment_amount * survival_prob * (1 - prob_default - prepay_rate)
            # Simple one-shot prepayment probability applied at the first scheduled payment
            expected_prepay = loan.remaining_balance * survival_prob * prepay_rate if i == 0 else 0.0
            expected_cashflows[week] += expected_payment + expected_prepay
            survival_prob *= (1 - prob_default - prepay_rate)
    return expected_cashflows


def aggregate_weekly_cashflows(loans, weeks=list(range(2, 20, 2))):
    weekly_cashflows = defaultdict(float)
    for week in weeks:
        for loan in loans:
            payment = loan.simulate_payment(week)
            weekly_cashflows[week] += payment
    return weekly_cashflows



def aggregate_cashflows(loans: List, waterfall=None) -> pd.DataFrame:
    """
    Aggregate actual recorded payments from each loan's payment_record and optionally include
    tranche-level residuals (e.g., Equity) from waterfall history.
    """
    from collections import defaultdict
    weekly_totals = defaultdict(float)

    for loan in loans:
        if hasattr(loan, "payment_record"):
            for week, amount in loan.payment_record.items():
                weekly_totals[week] += amount

    # Include tranche-level residuals (e.g., Equity) from waterfall history
    if waterfall is not None and hasattr(waterfall, "history"):
        for record in waterfall.history:
            for name, amount in record.get('tranche_payments', {}).items():
                if name == 'Equity':
                    weekly_totals[record['week']] += amount

    if not weekly_totals:
        return pd.DataFrame(columns=["week", "cashflow"])

    return pd.DataFrame(
        [{"week": week, "cashflow": amount} for week, amount in sorted(weekly_totals.items())]
    )



def simulate_tranche_waterfall(cashflow_df, tranche_structure=None):
    if tranche_structure is None:
        # Fallback tranche sizes derived from total expected cash over the horizon
        # (Only used when a structure isn't provided explicitly)
        total_loan_pool = float(cashflow_df["cashflow"].sum())
        tranche_structure = [
            {"name": "Senior", "principal": 0.50 * total_loan_pool, "rate": 0.06},
            {"name": "Mezzanine", "principal": 0.30 * total_loan_pool, "rate": 0.10},
            {"name": "Equity", "principal": 0.20 * total_loan_pool, "rate": 0.15},
        ]

    num_weeks = len(cashflow_df)
    weekly_cashflows = {tranche["name"]: [0] * num_weeks for tranche in tranche_structure}
    outstanding_balances = {tranche["name"]: tranche["principal"] for tranche in tranche_structure}
    rates = {tranche["name"]: tranche["rate"] for tranche in tranche_structure}

    for i, row in cashflow_df.iterrows():
        available_cash = row["cashflow"]

        for tranche in tranche_structure:
            name = tranche["name"]
            if name == "Equity":
                continue  # Equity is residual; pay it at the end from whatever is left
            balance = outstanding_balances[name]
            if balance <= 0:
                continue
            # Weekly interest payment on OUTSTANDING balance
            interest_due = balance * (rates[name] / 52)
            interest_paid = min(available_cash, interest_due)
            available_cash -= interest_paid
            weekly_cashflows[name][i] += interest_paid
            # Principal
            principal_due = balance
            principal_paid = min(available_cash, principal_due)
            available_cash -= principal_paid
            weekly_cashflows[name][i] += principal_paid
            outstanding_balances[name] -= principal_paid

        if available_cash > 0:
            weekly_cashflows["Equity"][i] += available_cash

    return weekly_cashflows



def print_summary_statistics(stats):
    print("\n====== Loan Portfolio Summary Statistics ======\n")
    
    print(f"Average Recovery Rate: {stats['average_recovery_rate']:.2%}")
    print(f"Total Late Fees Collected: ${stats['total_late_fees_collected']:,}")
    print(f"Average Effective APR: {stats['average_effective_apr']:.2%}\n")
    
    print("Default Rate by Credit Score Bucket:")
    for bucket, rate in stats['default_rate_by_bucket'].items():
        print(f"   • {bucket}: {rate:.2%}")
    
    print("\n=================================================\n")



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


def export_empirical_data(loans, path="empirical_data.csv"):
    rows = []
    for loan in loans:
        delay_list = getattr(loan, "payment_delays", [])
        for delay in delay_list:
            rows.append({
                "loan_id": loan.loan_id,
                "income": loan.income,
                "default_time": loan.default_time_week_continuous,
                "delay": delay,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def fit_income_distribution(data):
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    return {"distribution": "lognorm", "shape": shape, "scale": scale}

def fit_default_time_distribution(data):
    loc, scale = stats.expon.fit(data)
    return {"distribution": "expon", "loc": loc, "scale": scale}

def fit_delay_distribution(data):
    mu = np.mean(data)
    return {"distribution": "poisson", "mu": mu}
