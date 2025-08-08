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
from helpers import project_loan_cashflows, aggregate_weekly_cashflows, aggregate_cashflows, simulate_tranche_waterfall, print_summary_statistics, assign_correlated_defaults
from statistics import calculate_summary_statistics, generate_reports, generate_security_report
from tranche import Tranche, TrancheUnit
from valuation import compute_expected_tranche_npvs, compute_single_run_investor_metrics, compute_tranche_unit_npvs, calculate_npv_module
from waterfall import ReserveAccount, Waterfall
from copula import score_to_default_rate, generate_correlated_defaults

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
        week = 1
        while True:
            active_loans = 0
            total_weekly_payment = 0
            for loan in self.loans:
                if not loan.defaulted:  # modified for DataFrame or row['prepaid']  # modified for DataFrame or row['remaining_balance']  # modified for DataFrame == 0):
                    payment = loan.simulate_payment(week)
                    total_weekly_payment += payment
                    active_loans += 1

            if total_weekly_payment > 0:
                self.waterfall.apply_payments(total_weekly_payment, week)

            if active_loans == 0 or week > max_weeks:
                break
            week += 1

        self.weeks_run = week

    def get_cashflows_by_week(self) -> Dict[int, float]:
        cashflows = defaultdict(float)
        for week in range(2, self.weeks_run + 1, 2):
            for loan in self.loans:
                payment = loan.simulate_payment(week)
                cashflows[week] += payment

            weekly_total_payment = cashflows[week]




# ----------------------------
# Main Simulation
# ----------------------------
def simulate_loans(num_loans=20, seed=42):
    """
    Generate a realistic batch of loan data for simulation.
    
    Parameters:
        num_loans (int): Number of loans to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing the simulated loan data.
    """
    np.random.seed(seed)
    #random.seed(seed)

    # Order amount: log-normal for skewed purchase sizes
    order_amounts = np.random.lognormal(mean=3.5, sigma=0.5, size=num_loans).round(2)

    # Credit scores: Normal distribution clipped to FICO range
    credit_scores = np.random.normal(loc=660, scale=80, size=num_loans).clip(300, 850).round(0)

    # Income levels: Normally distributed with bounds
    incomes = np.random.normal(loc=50000, scale=20000, size=num_loans).clip(10000, 200000).round(0)

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
        'employment_status': employment_statuses
    })

    return df




def main():
    # 1️⃣ Generate loans
    loans_df = simulate_loans()

    valuation_loans = [
    ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score'])
        for _, row in loans_df.iterrows()
    ]

    assign_correlated_defaults(valuation_loans, rho=0.2, seed=42)

    df = pd.DataFrame([{
        "loan_id": row['loan_id'],
        "order_amount": row['order_amount'],
        "credit_score": row['credit_score'],
    } for _, row in loans_df.iterrows()])

    total_loan_pool = df["order_amount"].sum()

    # 2️⃣ Setup tranches and reserve
    tranches = [
        Tranche(name="Senior", principal=0.50 * total_loan_pool, priority=1),
        Tranche(name="Mezzanine", principal=0.30 * total_loan_pool, priority=2),
        Tranche(name="Equity", principal=0.20 * total_loan_pool, priority=3),
    ]
    reserve = ReserveAccount(target=0.05 * total_loan_pool)

    # 3️⃣ Valuation NPVs (model-predicted)
    valuation_loans = [ValuationLoan(row['loan_id'], row['order_amount'], row['credit_score']) 
                       for _, row in loans_df.iterrows()]
    true_npvs = compute_expected_tranche_npvs(tranches, reserve, valuation_loans)

    print("\n--- True Projected NPVs (Separated Valuation) ---")
    for t_name, npv in true_npvs.items():
        print(f"{t_name}: {npv:.2f}")

    # 4️⃣ Setup SPV and cashflow projection
    spv = SpecialPurposeVehicle(valuation_loans, tranches, reserve)

    # ✅ Aggregate all cashflows directly (no weeks argument needed), include waterfall for equity
    all_cashflows = aggregate_cashflows(spv.loans, waterfall=spv.waterfall)

    weeks = [2, 4, 6, 8]  # Standard biweekly schedule
    
    # 5️⃣ Projected tranche NPVs (initial pricing)
    projected_npvs = compute_expected_tranche_npvs(tranches, reserve, valuation_loans)
    print("\n--- Projected Tranche NPVs ---")
    for name, npv in projected_npvs.items():
        print(f"{name}: {npv:.2f}")

    # 6️⃣ Run the actual simulation
    spv.simulate_all_payments()
    metrics_df = compute_single_run_investor_metrics(spv.tranches)
    print("\n--- Tranche Investor Metrics ---")
    print(metrics_df.to_string(index=False))

    # 7️⃣ Generate loan-level and waterfall reports
    loan_df, tranche_df, reserve_df, waterfall_df = generate_reports(
        spv.loans, spv.tranches, spv.reserve, spv.waterfall
    )

    # 8️⃣ Calculate expected and realized loss
    expected_loss = sum(
        row["order_amount"] * score_to_default_rate(row["credit_score"])
        for _, row in loan_df.iterrows()
    )
    # Sum up equity payments from waterfall history
    equity_from_waterfall = sum(
        amt for record in spv.waterfall.history
        for name, amt in record.get('tranche_payments', {}).items() if name == 'Equity'
    )
    total_collected = all_cashflows["cashflow"].sum() + equity_from_waterfall
    realized_loss = loan_df["order_amount"].sum() - total_collected

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
    assign_correlated_defaults(loans, rho=0.2, seed=random.randint(0, 10000))
    total_loan_pool = loans_df['order_amount'].sum()

    # ✅ Simulate loan payments
    for week in [2, 4, 6, 8]:
        for loan in loans:
            loan.simulate_payment(week)

    # ✅ Aggregate cashflows after simulation
    cashflow_df = aggregate_cashflows(loans)

    tranche_structure = [
        {"name": "Senior", "principal": 0.50 * total_loan_pool, "rate": 0.05},
        {"name": "Mezzanine", "principal": 0.30 * total_loan_pool, "rate": 0.08},
        {"name": "Equity", "principal": 0.20 * total_loan_pool, "rate": 0.15},
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
    equity_losses = sum(df["NPV_Equity"] <= 0.01)

    print("\n--- Copula Impact Summary ---")
    print(f"Avg defaults per run: {default_counts.mean():.2f}")
    print(f"Runs with tail losses > $2000: {tail_loss_runs}/10")
    print(f"Equity tranche received no meaningful payout in {equity_losses}/10 runs")
    print("\n")



