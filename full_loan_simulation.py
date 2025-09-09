import random
from collections import defaultdict
import pandas as pd
from typing import List, Dict
import numpy_financial as npf
import numpy as np

from typing import List, Dict
from collections import defaultdict
import math
from copy import deepcopy

from loan import Loan, ValuationLoan
from helpers import (project_loan_cashflows, aggregate_weekly_cashflows, aggregate_cashflows, simulate_tranche_waterfall, 
                    print_summary_statistics, assign_correlated_defaults, fit_income_distribution,
                    fit_default_time_distribution, fit_delay_distribution)
from statistics import calculate_summary_statistics, generate_reports, generate_security_report
from tranche import Tranche, TrancheUnit
from valuation import compute_expected_tranche_npvs, compute_single_run_investor_metrics, compute_tranche_unit_npvs, calculate_npv_module
from waterfall import ReserveAccount, Waterfall
from copula import generate_correlated_defaults
from hazard_models import score_to_default_rate
from hazard_models import cox_hazard_rate
from credit_state import simulate_credit_transition

import numpy as np
import numpy_financial as npf

from scipy.stats import norm
from scipy.stats import lognorm

def annual_to_horizon_pd(pd_annual: float, horizon_years: float) -> float:
    """Convert annual PD to a horizon PD assuming independent hazard over time."""
    return 1.0 - (1.0 - pd_annual) ** horizon_years

def weekly_hazard_from_horizon_pd(horizon_pd: float, horizon_weeks: int) -> float:
    """
    Solve for a constant weekly hazard p such that:
    1 - (1 - p)**horizon_weeks = horizon_pd  =>  p = 1 - (1 - horizon_pd)**(1/horizon_weeks)
    """
    horizon_pd = float(np.clip(horizon_pd, 1e-12, 1 - 1e-12))
    return 1.0 - (1.0 - horizon_pd) ** (1.0 / horizon_weeks)

def assign_correlated_default_times_hybrid(
    loans: List[ValuationLoan],
    rho: float = 0.07,
    seed: int = 42,
    due_weeks: tuple = (2, 4, 6, 8),
    horizon_weeks: int = 8,
    due_hazard_multiplier: float = 1.5,
) -> None:
    rng = np.random.default_rng(seed)
    n = len(loans)
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1).")

    # Common/Systematic factor and idiosyncratic shocks
    M = rng.standard_normal()
    eps = rng.standard_normal(n)
    z = np.sqrt(rho) * M + np.sqrt(1.0 - rho) * eps
    U = norm.cdf(z)  # Copula uniforms in (0,1)

    # For placing the event within the selected week
    U_within = rng.random(n)

    horizon_years = horizon_weeks / 52.0
    due_set = set(due_weeks)

    # Ensure the multiplier keeps hazards positive on "other" weeks
    m = float(due_hazard_multiplier)

    if not (0.0 < m < 2.0):
        raise ValueError("due_hazard_multiplier must be in (0,2) so that p_other remains positive.")

    for i, loan in enumerate(loans):
        # Use Cox model for hazard estimation instead of static PD
        hazard_rate = float(cox_hazard_rate(getattr(loan, "credit_score", None)))
        pd_hor = annual_to_horizon_pd(hazard_rate, horizon_years)
        p_base = weekly_hazard_from_horizon_pd(pd_hor, horizon_weeks)

        # Two-level weekly hazards; clip to a sensible range
        p_due = np.clip(m * p_base, 1e-9, 1 - 1e-6)
        p_other = np.clip((2.0 - m) * p_base, 1e-9, 1 - 1e-6)

        weekly_haz = []
        for w in range(1, horizon_weeks + 1):
            weekly_haz.append(p_due if w in due_set else p_other)

        # Discrete inverse-transform using U[i]
        S = 1.0  # survival prior to week 1
        default_week = None
        for w, p_w in enumerate(weekly_haz, start=1):
            if U[i] <= (1.0 - (S * (1.0 - p_w))):
                default_week = w
                break
            S *= (1.0 - p_w)

        if default_week is None:
            # Survived the horizon
            loan.default_time_week_continuous = float("inf")
            loan.default_week = None
            loan.default_scheduled_week_to_pay = None
            continue

        # Place the default uniformly within the selected week
        t_cont = (default_week - 1) + U_within[i]
        loan.default_time_week_continuous = float(t_cont)
        loan.default_week = int(default_week)

        loan.defaulted = True

        # Simulate a credit rating downgrade or transition
        if hasattr(loan, "credit_state") and loan.credit_state != "D":
            new_state = simulate_credit_transition(loan.credit_state)
            loan.credit_state = new_state

        # For BNPL payment logic: snap to the *next* payment week >= t_cont
        next_pay_weeks = [w for w in due_weeks if w >= t_cont - 1e-9]
        loan.default_scheduled_week_to_pay = int(next_pay_weeks[0]) if next_pay_weeks else None



# Local aggregator: build weekly pool cashflows from each loan's recorded payments
def aggregate_cashflows_local(loans, horizon_weeks: int = 8) -> pd.DataFrame:
    from collections import defaultdict
    totals = defaultdict(float)
    for loan in loans:
        pr = getattr(loan, "payment_record", {})
        if isinstance(pr, dict):
            for w, amt in pr.items():
                try:
                    w_int = int(w)
                except Exception:
                    continue
                totals[w_int] += float(amt)
    # Ensure biweekly schedule rows exist (2,4,6,8) with zeros if missing
    schedule = [2, 4, 6, 8] if horizon_weeks >= 8 else [w for w in [2,4,6,8] if w <= horizon_weeks]
    rows = []
    for w in schedule:
        rows.append((w, float(totals.get(w, 0.0))))
    # Also include any extra weeks found in totals (sorted) that are not in schedule
    extra_weeks = sorted([w for w in totals.keys() if w not in schedule])
    for w in extra_weeks:
        rows.append((w, float(totals[w])))
    if not rows:
        return pd.DataFrame({"week": [], "cashflow": [], "total_payment": [], "amount": []})
    rows = sorted(rows, key=lambda x: x[0])
    df = pd.DataFrame(rows, columns=["week", "cashflow"])
    # Provide redundant column names for downstream compatibility
    df["total_payment"] = df["cashflow"]
    df["amount"] = df["cashflow"]
    return df

def monte_carlo_tranche_npvs(loans_df, tranche_structure, rho=0.07, discount_rates=None, n_paths=200):
    """Price tranches by Monte Carlo using the SAME mechanics as realized path."""
    from helpers import aggregate_cashflows, simulate_tranche_waterfall, assign_correlated_defaults
    from loan import ValuationLoan
    import random

    if discount_rates is None:
        discount_rates = {"Senior": 0.05, "Mezzanine": 0.08, "Equity": 0.15}

    tranche_names = [t["name"] for t in tranche_structure]
    sums = {name: 0.0 for name in tranche_names}

    for _ in range(n_paths):
        loans = [
            ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score'])
            for _, row in loans_df.iterrows()
        ]
        
        assign_correlated_default_times_hybrid(loans, rho=rho, seed=random.randint(0, 10**9), due_hazard_multiplier=1.5)

        for week in [2, 4, 6, 8]:
            for loan in loans:
                loan.simulate_payment(week)
        cashflow_df = aggregate_cashflows_local(loans)
        tranche_cf = simulate_tranche_waterfall(cashflow_df, tranche_structure)
        for name, cf in tranche_cf.items():
            sums[name] += float(npf.npv(discount_rates[name]/52.0, cf))

    return {name: sums[name] / n_paths for name in tranche_names}


# ----------------------------
# Special Purpose Vehicle (SPV) Class
# ----------------------------
class SpecialPurposeVehicle:
    def __init__(self, loans: List[Loan], tranches: List[Tranche], reserve: ReserveAccount):
        self.loans = loans
        self.weeks_run = 0
        self.tranches = tranches
        self.reserve = reserve
        self.waterfall = Waterfall(tranches, reserve, self.total_loan_amount())

    def total_loan_amount(self):
        return sum(loan.order_amount for loan in self.loans)

    def simulate_all_payments(self, max_weeks: int = 52):
        """Simulate loan payments only on due weeks, then aggregate and run the same
        waterfall used in MC pricing. Persist per-tranche cashflow arrays so
        investor metrics (IRR/CoC) work for the single SPV path.
        """
        due_weeks = (2, 4, 6, 8)
        self.weeks_run = 0

        # 1) Generate loan-level payments on the due schedule
        for week in due_weeks:
            for loan in self.loans:
                loan.simulate_payment(week)
            self.weeks_run = week

        # 2) Aggregate pool cashflows exactly like MC
        cashflow_df = aggregate_cashflows_local(self.loans)

        # 3) Build a tranche structure (name/principal/rate) from current tranches
        tranche_structure = [
            {
                "name": t.name,
                "principal": float(getattr(t, "principal", 0.0)),
                "rate": 0.05 if t.name == "Senior" else (0.08 if t.name == "Mezzanine" else 0.15),
            }
            for t in self.tranches
        ]

        # 4) Run the same waterfall used in MC to get weekly tranche cashflows
        tranche_cf = simulate_tranche_waterfall(cashflow_df, tranche_structure)

        # 5) Persist per-tranche arrays for downstream IRR/CoC and reporting
        for t in self.tranches:
            cf = list(tranche_cf.get(t.name, []))
            # Store full history so compute_single_run_investor_metrics can read it
            setattr(t, "cashflows", cf)
            t.total_received = float(sum(cf))

    def get_cashflows_by_week(self) -> Dict[int, float]:
        """Return a {week: cashflow} dict using recorded loan payment_history.
        Does not call simulate_payment again (avoids double-posting)."""
        df = aggregate_cashflows_local(self.loans)
        return {int(w): float(a) for w, a in zip(df["week"].tolist(), df["cashflow"].tolist())}



# ----------------------------
# Main Simulation
# ----------------------------
def simulate_loans(num_loans=20):
    """
    Generate a realistic batch of loan data for simulation.
    
    Parameters:
        num_loans (int): Number of loans to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing the simulated loan data.
    """
    #np.random.seed(seed)
    #random.seed(seed)

    # Order amount: log-normal for skewed purchase sizes
    order_amounts = np.random.lognormal(mean=3.5, sigma=0.5, size=num_loans).round(2)

    # Credit scores: Normal distribution clipped to FICO range
    credit_scores = np.random.normal(loc=660, scale=80, size=num_loans).clip(300, 850).round(0)

    empirical_income_data = np.random.lognormal(mean=10.5, sigma=0.5, size=1000)

    # Income levels: Normally distributed with bounds
    params = fit_income_distribution(empirical_income_data)
    incomes = lognorm.rvs(s=params["shape"], scale=params["scale"], size=num_loans)
    # For demonstration, mock empirical default times and delays if not present
    empirical_default_times = np.random.exponential(scale=2.0, size=1000)
    empirical_delays = np.random.poisson(lam=1.0, size=1000)
    # Sample default times from exponential fit
    default_params = fit_default_time_distribution(empirical_default_times)
    default_times = np.random.exponential(scale=default_params["scale"], size=num_loans)
    # Sample payment delays from Poisson fit
    delay_params = fit_delay_distribution(empirical_delays)
    payment_delays = np.random.poisson(lam=delay_params["mu"], size=num_loans)

    # Employment status: Categorical distribution
    employment_statuses = np.random.choice(
        ['employed', 'unemployed', 'student', 'self-employed'],
        size=num_loans,
        p=[0.6, 0.15, 0.1, 0.15]
    )

    # Credit quality labels based on score
    credit_quality = pd.cut(
        credit_scores,
        bins=[299, 600, 700, 850],
        labels=["bad", "fair", "good"]
    )

    # Combine into a DataFrame
    df = pd.DataFrame({
        'loan_id': range(1, num_loans + 1),
        'order_amount': order_amounts,
        'credit_score': credit_scores,
        'credit_quality': credit_quality,
        'income': incomes,
        'employment_status': employment_statuses,
        'default_time': default_times,
        'simulated_delay': payment_delays
    })

    return df


def main():
    # 1️⃣ Generate loans
    loans_df = simulate_loans()

    valuation_loans = []
    for _, row in loans_df.iterrows():
        loan = ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score'])
        loan.income = row['income']
        valuation_loans.append(loan)

    assign_correlated_default_times_hybrid(valuation_loans, rho=0.7, seed=42, due_hazard_multiplier=1.5)

    df = pd.DataFrame([{
        "loan_id": row['loan_id'],
        "order_amount": row['order_amount'],
        "credit_score": row['credit_score'],
    } for _, row in loans_df.iterrows()])

    total_loan_pool = df["order_amount"].sum()

    tranche_structure = [
        {"name": "Senior", "principal": 0.40 * total_loan_pool, "rate": 0.05},
        {"name": "Mezzanine", "principal": 0.30 * total_loan_pool, "rate": 0.08},
        {"name": "Equity", "principal": 0.30 * total_loan_pool, "rate": 0.15},
    ]
    discount_rates = {"Senior": 0.05, "Mezzanine": 0.08, "Equity": 0.15}

    true_npvs = monte_carlo_tranche_npvs(loans_df, tranche_structure, rho=0.07, discount_rates=discount_rates, n_paths=200)

    print("\n--- True Projected NPVs (Separated Valuation) ---")
    for t_name, npv in true_npvs.items():
        print(f"{t_name}: {npv:.2f}")

    tranches = [
        Tranche(name="Senior", principal=tranche_structure[0]["principal"], priority=1),
        Tranche(name="Mezzanine", principal=tranche_structure[1]["principal"], priority=2),
        Tranche(name="Equity", principal=tranche_structure[2]["principal"], priority=3),
    ]
    reserve = ReserveAccount(target=0.0)

    # 3️⃣ Valuation NPVs (model-predicted)
    #valuation_loans = [ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score']) 
    #                   for _, row in loans_df.iterrows()]

    projected_npvs = monte_carlo_tranche_npvs(loans_df, tranche_structure, rho=0.07, discount_rates=discount_rates, n_paths=200)
    print("\n--- Projected Tranche NPVs ---")
    for name, npv in projected_npvs.items():
        print(f"{name}: {npv:.2f}")

    # 4️⃣ Setup SPV and cashflow projection
    spv = SpecialPurposeVehicle(valuation_loans, tranches, reserve)

    weeks = [2, 4, 6, 8]  # Standard biweekly schedule

    # 6️⃣ Run the actual simulation
    spv.simulate_all_payments()


    # ✅ Aggregate all cashflows AFTER the simulation has generated them
    metrics_df = compute_single_run_investor_metrics(spv.tranches)
    print("\n--- Tranche Investor Metrics ---")
    print(metrics_df.to_string(index=False))

    # 7️⃣ Generate loan-level and waterfall reports
    loan_df, tranche_df, reserve_df, waterfall_df = generate_reports(
        spv.loans, spv.tranches, spv.reserve, spv.waterfall
    )

    # 8️⃣ Calculate expected and realized loss (horizon-adjusted PD × LGD over 8 weeks)
    # Expected loss aligned with valuation/statistics (PD_horizon * LGD over 8 weeks)
    lgd_assumption = 0.70
    horizon_years = 8.0 / 52.0
    expected_loss = 0.0
    for _, row in loan_df.iterrows():
        pd_ann = float(score_to_default_rate(row["credit_score"]))
        pd_ann = max(0.0, min(pd_ann, 1.0))
        pd_h = 1.0 - (1.0 - pd_ann) ** horizon_years
        expected_loss += float(row["order_amount"]) * pd_h * lgd_assumption

    # Sum up equity payments from waterfall history
    all_cashflows = aggregate_cashflows_local(spv.loans)
    total_collected = float(all_cashflows["cashflow"].sum())
    realized_loss = float(loan_df["order_amount"].sum()) - total_collected

    # 9️⃣ Run NPV calculation for entire loan pool
    npv_output = calculate_npv_module(
        all_cashflows,
        loans=spv.loans,
        annual_discount_rates=[0.05, 0.10, 0.15],
        expected_loss=expected_loss,
        realized_loss=realized_loss
    )

    print("\n=== Combined Loan Pool NPV Analysis ===")
    for rate, value in npv_output["NPV_Results"].items():
        print(f"Discount Rate {rate}: {value:.2f}")

    loss = npv_output["Loss_Comparison"]
    if loss:
        print(f"Expected Loss: {loss['expected_loss']:.2f}")
        print(f"Realized Loss: {loss['realized_loss']:.2f}")
        print(f"Difference: {loss['difference']:.2f}")

    summary_stats = calculate_summary_statistics(spv.loans)
    def print_summary_statistics(stats):
        print("\n====== Loan Portfolio Summary Statistics ======\n")
        
        print(f"Average Recovery Rate: {stats['average_recovery_rate']:.2%}")
        print(f"Total Late Fees Collected: ${stats['total_late_fees_collected']:,}")
        print(f"Average Effective APR: {stats['average_effective_apr']:.2%}\n")
        
        print("Default Rate by Credit Score Bucket:")
        for bucket, rate in stats['default_rate_by_bucket'].items():
            print(f"   • {bucket}: {rate:.2%}")
        
        print("\n=================================================\n")

    print_summary_statistics(summary_stats)


if __name__ == "__main__":
  main()


def run_simulation_summary(num_loans=500, discount_rates=None):
    if discount_rates is None:
        discount_rates = {
            "Senior": 0.05,
            "Mezzanine": 0.08,
            "Equity": 0.15,
        }

    loans_df = simulate_loans(num_loans)
    loans = [
        ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score'])
        for _, row in loans_df.iterrows()
    ]

    from helpers import assign_correlated_defaults

    assign_correlated_default_times_hybrid(loans, rho=0.07, seed=random.randint(0, 10000), due_hazard_multiplier=1.5)
    
    total_loan_pool = loans_df['order_amount'].sum()

    # ✅ Simulate loan payments
    for week in [2, 4, 6, 8]:
        for loan in loans:
            loan.simulate_payment(week)

    # ✅ Aggregate cashflows after simulation
    cashflow_df = aggregate_cashflows_local(loans)

    tranche_structure = [
        {"name": "Senior", "principal": 0.40 * total_loan_pool, "rate": 0.05},
        {"name": "Mezzanine", "principal": 0.30 * total_loan_pool, "rate": 0.08},
        {"name": "Equity", "principal": 0.30 * total_loan_pool, "rate": 0.15},
    ]

    tranche_cashflows = simulate_tranche_waterfall(cashflow_df, tranche_structure)

    npvs_by_tranche = {
        tranche_name: npf.npv(discount_rates[tranche_name] / 52, cashflows)
        for tranche_name, cashflows in tranche_cashflows.items()
    }

    num_defaults = sum(1 for loan in loans if loan.defaulted)
    total_losses = sum(loan.order_amount for loan in loans if loan.defaulted)

    return {
        "num_defaults": num_defaults,
        "total_losses": total_losses,
        "npvs_by_tranche": npvs_by_tranche,
        "cashflow_df": cashflow_df,
    }


if __name__ == "__main__":
    results = []

    # Simulation loop (Monte Carlo)
    for i in range(10):
        summary = run_simulation_summary()
        results.append(summary)

    df = pd.DataFrame(results)

    # Flatten npvs_by_tranche dict into separate columns
    # Converts the dictionary column into separate NPV_Senior, NPV_Mezzanine, etc columns
    npv_df = df["npvs_by_tranche"].apply(pd.Series)
    npv_df.columns = [f"NPV_{col}" for col in npv_df.columns]

    # Drop the original messy column and join clean one
    df = df.drop(columns=["npvs_by_tranche"]).join(npv_df)

    # Round all numerical columns
    df = df.round(2)
    print("\n--- Simulation Summary (10 runs) ---")
    summary_df = df.drop(columns=['cashflow_df'], errors='ignore')
    print(summary_df.to_string(index=False))

    default_counts = df["num_defaults"]
    tail_loss_runs = sum(df["total_losses"] > 2000)
    threshold = 1.0  # treat <= $1 as negligible; tweak as you like
    paid_runs = int((df["NPV_Equity"] > threshold).sum())
    zero_runs = len(df) - paid_runs

    print("\n--- Copula Impact Summary ---")
    print(f"Avg defaults per run: {default_counts.mean():.2f}")
    print(f"Runs with tail losses > $2000: {tail_loss_runs}/{len(df)}")
    print(f"Equity paid meaningfully (> ${threshold:.0f}) in {paid_runs}/{len(df)} runs; "f"negligible in {zero_runs}/{len(df)} runs.")
    print("\n")

    # --- Risk Metrics (MC) ---
    equity_wipe_threshold = 1.0  # treat <= $1 NPV as wiped; tweak if needed

    # Probability equity is wiped
    p_equity_wiped = float((df["NPV_Equity"] <= equity_wipe_threshold).mean())

    # Loss percentiles (P50 / P95) and Expected Shortfall at 95%
    losses = df["total_losses"].to_numpy(dtype=float)
    p50_loss = float(np.percentile(losses, 50))
    p95_loss = float(np.percentile(losses, 95))
    es95_loss = float(losses[losses >= p95_loss].mean()) if (losses >= p95_loss).any() else p95_loss

    print("--- Risk Metrics ---")
    print(f"P(Equity wiped) (NPV_Equity <= ${equity_wipe_threshold:.0f}): {p_equity_wiped:.2%}")
    print(f"Portfolio loss P50: ${p50_loss:,.2f}")
    print(f"Portfolio loss P95: ${p95_loss:,.2f}")
    print(f"Expected shortfall (>= P95): ${es95_loss:,.2f}")
    print("\n")
