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




# ----------------------------
# Loan Class
# ----------------------------
class Loan:
    def __init__(self, loan_id, order_amount, credit_score, late_fee=5):
        self.loan_id = loan_id
        self.order_amount = order_amount
        self.credit_score = credit_score
        self.installment_amount = order_amount / 4
        self.remaining_balance = order_amount
        self.defaulted = False
        self.prepaid = False
        self.late_weeks = []
        self.paid_weeks = []
        self.payment_record = {}
        self.expected_payment_schedule = [2, 4, 6, 8]
        self.due_stack = []
        self.late_fee = late_fee
        self.total_late_fees_paid = 0
        self.missed_payment_count = 0

    def simulate_payment(self, week):
        if self.defaulted or self.prepaid:
            return 0

        if not self.prepaid and week in self.expected_payment_schedule and random.random() < 0.05:
            payment = self.remaining_balance
            self.remaining_balance = 0
            self.prepaid = True
            self.payment_record[week] = payment
            return payment

        if week in self.expected_payment_schedule:
            self.due_stack.append((week, self.installment_amount, 0))

        total_payment = 0
        still_due = []

        if self.credit_score < 600:
            miss_payment_chance = 0.12
            default_threshold = 4
        elif self.credit_score < 700:
            miss_payment_chance = 0.07
            default_threshold = 5
        else:
            miss_payment_chance = 0.02
            default_threshold = 6

        for due_week, amount_due, late_fee in self.due_stack:
            if (week - due_week) >= 2 and late_fee == 0:
                proposed_fee = min(7, 0.25 * self.order_amount - self.total_late_fees_paid)
                late_fee = max(0, proposed_fee)

            total_due = amount_due + late_fee
            will_miss_payment = random.random() < miss_payment_chance

            if will_miss_payment or self.remaining_balance < total_due:
                still_due.append((due_week, amount_due, late_fee))
                if week == due_week and will_miss_payment:
                    self.late_weeks.append(due_week)
                    self.missed_payment_count += 1
                    if self.missed_payment_count >= default_threshold:
                        self.defaulted = True

                        recovery_rate = 0.30  # 30% recovery of remaining balance
                        recovery_amount = self.remaining_balance * recovery_rate
    
                        self.remaining_balance = 0  # Loan is written off
                        self.payment_record[week] = self.payment_record.get(week, 0) + recovery_amount
    
                        return total_payment + recovery_amount

            else:
                self.remaining_balance -= total_due
                total_payment += total_due
                self.total_late_fees_paid += late_fee
                self.paid_weeks.append(week)
                self.payment_record[week] = self.payment_record.get(week, 0) + total_due

        self.due_stack = still_due
        return total_payment if not self.defaulted else 0

# ----------------------------
# Tranche Class
# ----------------------------
class Tranche:
    def __init__(self, name, principal, priority, unit_size=100):
        self.name = name
        self.principal = principal
        self.priority = priority
        self.unit_size = unit_size
        self.units = self._create_units()
        self.total_paid = 0

    def _create_units(self):
        num_units = int(self.principal // self.unit_size)
        return [TrancheUnit(self.unit_size) for _ in range(num_units)]

    def apply_payment(self, amount, week=None):
      remaining = amount
      for unit in self.units:
        if remaining <= 0:
          break
        remaining = unit.apply_payment(remaining, week=week)
      paid_now = amount - remaining
      self.total_paid += paid_now
      return paid_now

    def is_fully_paid(self):
        return all(unit.paid for unit in self.units)

# ----------------------------
# Tranche Unit Class
# ----------------------------
class TrancheUnit:
    def __init__(self, face_value=100):
        self.face_value = face_value
        self.remaining_principal = face_value
        self.paid = False
        self.cashflow_history = {}

    def apply_payment(self, amount, week=None):
        if self.paid:
            return amount  # already paid off, return full amount

        payment = min(amount, self.remaining_principal)
        self.remaining_principal -= payment
        if self.remaining_principal == 0:
          self.paid = True

        if week is not None:
          self.cashflow_history[week] = self.cashflow_history.get(week, 0) + payment

        return amount - payment

# ----------------------------
# Reserve Account Class
# ----------------------------
class ReserveAccount:
    def __init__(self, target = 100):
        self.balance = 0
        self.history = {}
        self.target = target

    def top_up(self, amount, week):
        self.balance += amount
        self.history[week] = self.balance

    def draw(self, shortfall, week):
        draw_amount = min(self.balance, shortfall)
        self.balance -= draw_amount
        self.history[week] = self.balance
        return draw_amount

# ----------------------------
# Waterfall Class
# ----------------------------
class Waterfall:
    def __init__(self, tranches, reserve_account, total_loan_pool):
        self.tranches = sorted(tranches, key=lambda x: x.priority)
        self.reserve = reserve_account
        self.history = []

        # Validation logic
        tolerance = 0.01  # 1 cent
        total_tranche_size = sum(tranche.principal for tranche in self.tranches)

        if abs(total_tranche_size - total_loan_pool) > tolerance:
            raise ValueError(
                f"Mismatch: Tranche total (${total_tranche_size}) vs. loan pool (${total_loan_pool})"
            )

    def apply_payments(self, cash, week=None):
        record = {
            'week': week,
            'starting_cash': cash,
            'reserve_draw': 0,
            'reserve_balance_start': self.reserve.balance,
            'tranche_payments': {},
            'leftover_cash': 0,
            'reserve_top_up': 0,
            'reserve_balance_end': 0,
        }

        available_cash = cash

        # Step 1: Draw from reserve if needed
        if available_cash < 0:
            shortfall = abs(available_cash)
            draw_amount = min(self.reserve.balance, shortfall)
            self.reserve.balance -= draw_amount
            available_cash += draw_amount
            record['reserve_draw'] = draw_amount

        # Step 2: Sequentially pay tranches
        for tranche in self.tranches:
            if available_cash <= 0:
                break
            paid = tranche.apply_payment(available_cash, week=week)
            available_cash -= paid
            record['tranche_payments'][tranche.name] = paid

        if available_cash > 0:
            record['tranche_payments']['Equity'] = (record['tranche_payments'].get('Equity', 0) + available_cash)
            available_cash = 0

        # Step 3: Top up reserve if there's leftover cash
        reserve_top_up = min(available_cash, self.reserve.target - self.reserve.balance)
        self.reserve.balance += reserve_top_up
        available_cash -= reserve_top_up
        record['reserve_top_up'] = reserve_top_up

        # Step 4: Record remaining values
        record['leftover_cash'] = available_cash
        record['reserve_balance_end'] = self.reserve.balance

        self.history.append(record)
        return record['tranche_payments']

# ----------------------------
# Default Probability Helper
# ----------------------------
def score_to_default_rate(score: int) -> float:
    if score > 700:
        return 0.02  # Higher risk
    elif score > 650:
        return 0.07  # Moderate risk
    else:
        return 0.12  # Lower risk

def project_loan_cashflows(loans, weeks, prepay_rate=0.01):
    expected_cashflows = {week: 0 for week in weeks}
    for loan in loans:  # loans is a list of ValuationLoan objects
        prob_default = score_to_default_rate(loan.credit_score)
        survival_prob = 1.0
    for i, week in enumerate(loan.expected_payment_schedule):
        expected_payment = loan.installment_amount * survival_prob * (1 - prob_default - prepay_rate)
        expected_prepay = loan.remaining_balance * survival_prob * prepay_rate if i == 0 else 0
        expected_cashflows[week] += expected_payment + expected_prepay
        survival_prob *= (1 - prob_default - prepay_rate)
    return expected_cashflows


# --------------------------
# Tranche Reporting Function
# --------------------------
def generate_reports(loans, tranches, reserve, waterfall):
    # -------------------------------
    # Loan-Level Summary
    # -------------------------------
    loan_summaries = []
    for loan in loans:  # modified for DataFrame structure
        status = "active"
        if loan.defaulted:  # modified for DataFrame:
            status = "defaulted"
        elif loan.prepaid:  # modified for DataFrame:
            status = "prepaid"
        elif loan.remaining_balance:  # modified for DataFrame <= 0:
            status = "paid"


        loan_summaries.append({
            "loan_id": loan.loan_id,  # modified for DataFrame,
            "order_amount": loan.order_amount,  # modified for DataFrame,
            "credit_score": loan.credit_score,  # modified for DataFrame,
            "status": status,
            "remaining_balance": round(loan.remaining_balance),  # modified for DataFrame, 2),
            "missed_payments": len(set(loan.late_weeks)),
            "late_weeks": sorted(set(loan.late_weeks)),
            "paid_weeks": sorted(set(loan.paid_weeks)),
            "total_late_fees_paid": round(loan.total_late_fees_paid, 2),
            "payment_record": loan.payment_record
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
# ----------------------------0
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



class ValuationLoan:
    def __init__(self, loan_id, order_amount, credit_score):
        self.loan_id = loan_id
        self.order_amount = order_amount
        self.credit_score = credit_score

        self.remaining_balance = order_amount
        self.installment_amount = order_amount / 4
        self.expected_payment_schedule = [2, 4, 6, 8]

        self.defaulted = False
        self.prepaid = False
        self.late_weeks = []
        self.paid_weeks = []
        self.total_late_fees_paid = 0
        self.payment_record = {}  # ✅ use dict for easier aggregation

    def expected_cashflows(self, prepay_rate=0.01):
        prob_default = score_to_default_rate(self.credit_score)
        survival_prob = 1.0
        cashflows = {}

        for i, week in enumerate(self.expected_payment_schedule):
            expected_payment = self.installment_amount * survival_prob * (1 - prob_default - prepay_rate)
            expected_prepay = self.remaining_balance * survival_prob * prepay_rate if i == 0 else 0
            cashflows[week] = expected_payment + expected_prepay
            survival_prob *= (1 - prob_default - prepay_rate)

        return cashflows

    def simulate_payment(self, week):
        if self.defaulted or self.prepaid or self.remaining_balance <= 0:
            return 0

        # Default logic
        prob_default = score_to_default_rate(self.credit_score)
        if random.random() < prob_default:
            self.defaulted = True
            
            recovery_rate = 0.30
            recovery_amount = self.remaining_balance * recovery_rate
            self.remaining_balance = 0
            self.payment_record[week] = recovery_amount
    
            return recovery_amount

        # Prepayment logic
        if random.random() < 0.08:
            self.prepaid = True
            payment = self.remaining_balance
            self.remaining_balance = 0
            self.paid_weeks.append(week)
            self.payment_record[week] = payment
            return payment

        # Late payment logic
        if random.random() < 0.2:
            self.late_weeks.append(week)
            self.total_late_fees_paid += 5
            self.payment_record[week] = 0
            return 0

        # ✅ Normal installment payment
        payment = min(self.installment_amount, self.remaining_balance)
        self.remaining_balance -= payment
        self.paid_weeks.append(week)
        self.payment_record[week] = payment
        return payment


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


def calculate_summary_statistics(loans: List[Loan]):
    stats = {}

    # Average recovery rate
    total_principal = sum(loan.order_amount for loan in loans)
    total_received = sum(sum(loan.payment_record.values()) for loan in loans)
    stats['average_recovery_rate'] = total_received / total_principal if total_principal > 0 else 0

    # Default rate by credit score bucket
    buckets = {'<600': [], '600-699': [], '>=700': []}
    for loan in loans:
        if loan.credit_score < 600:
            buckets['<600'].append(loan)
        elif loan.credit_score < 700:
            buckets['600-699'].append(loan)
        else:
            buckets['>=700'].append(loan)

    stats['default_rate_by_bucket'] = {
        bucket: (sum(1 for l in loans_list if l.defaulted) / len(loans_list) if loans_list else 0)
        for bucket, loans_list in buckets.items()
    }

    # Total interest/late fees collected
    stats['total_late_fees_collected'] = sum(loan.total_late_fees_paid for loan in loans)

    # Effective APR (approximate)
    aprs = []
    for loan in loans:
        if loan.payment_record:
            total_paid = sum(loan.payment_record.values())
            apr = ((total_paid / loan.order_amount) - 1) * (52 / 8)  # 8 weeks ~ 2 months
            aprs.append(apr)
    stats['average_effective_apr'] = sum(aprs) / len(aprs) if aprs else 0

    return stats


# ----------------------------
# Main Simulation
# ----------------------------
def simulate_loans(num_loans=100, seed=42):
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
    credit_scores = np.random.normal(loc=680, scale=50, size=num_loans).clip(300, 850).round(0)

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


def aggregate_weekly_cashflows(loans, weeks=list(range(2, 20, 2))):
    weekly_cashflows = defaultdict(float)
    for week in weeks:
        for _, row in loans_df.iterrows():  # modified for DataFrame structure
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
    """
    Simulates a tranche-based cashflow waterfall for a CDO.

    Parameters:
        cashflow_df (pd.DataFrame): Must contain 'Week' and 'Net_Collections' columns.
        tranche_structure (list of dict): Each dict defines a tranche:
            [
                {"name": "Senior", "principal": 1000, "rate": 0.06},
                {"name": "Mezzanine", "principal": 500, "rate": 0.10},
                {"name": "Equity", "principal": 0, "rate": 0.0}
            ]

    Returns:
        dict: Weekly cashflows allocated to each tranche.
    """
    if tranche_structure is None:
        tranche_structure = [
            {"name": "Senior", "principal": 1000, "rate": 0.06},
            {"name": "Mezzanine", "principal": 500, "rate": 0.10},
            {"name": "Equity", "principal": 0.20 * total_loan_pool, "rate": 0.15}
        ]

    num_weeks = len(cashflow_df)
    weekly_cashflows = {tranche["name"]: [0] * num_weeks for tranche in tranche_structure}
    outstanding_balances = {tranche["name"]: tranche["principal"] for tranche in tranche_structure}
    rates = {tranche["name"]: tranche["rate"] for tranche in tranche_structure}

    for i, row in cashflow_df.iterrows():
        available_cash = row["cashflow"]

        for tranche in tranche_structure:
            name = tranche["name"]
            balance = outstanding_balances[name]

            if balance <= 0:
                continue

            # Weekly interest payment
            interest_due = balance * (rates[name] / 52)
            interest_paid = min(available_cash, interest_due)
            available_cash -= interest_paid
            weekly_cashflows[name][i] += interest_paid

            # Principal payment
            principal_due = balance
            principal_paid = min(available_cash, principal_due)
            available_cash -= principal_paid
            weekly_cashflows[name][i] += principal_paid
            outstanding_balances[name] -= principal_paid

        # Anything left over goes to equity
        if available_cash > 0:
            weekly_cashflows["Equity"][i] += available_cash

    return weekly_cashflows


def main():
    # 1️⃣ Generate loans
    loans_df = simulate_loans()

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


def run_simulation_summary(num_loans=20, discount_rates=None):
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
    total_loan_pool = loans_df['order_amount'].sum()

    # ✅ Simulate loan payments
    for week in [2, 4, 6, 8]:
        for loan in loans:
            loan.simulate_payment(week)

    # ✅ Aggregate cashflows after simulation
    cashflow_df = aggregate_cashflows(loans)

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
    total_losses = sum(loan.remaining_balance for loan in loans if loan.defaulted)

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




