from hazard_models import score_to_default_rate
import random
import numpy as np
from scipy.stats import norm

# ----------------------------
# Loan Class
# ----------------------------
class Loan:
    def __init__(self, loan_id, order_amount, credit_score, late_fee=5, externally_defaulted=None):
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
        self.externally_defaulted = externally_defaulted

    def set_default(self):
        self.defaulted = True
        self.remaining_balance = 0

    def simulate_path_payment(self, week: int) -> float:
        # Idempotency: don't double-post cashflows for the same week
        if week in self.payment_record:
            return 0.0

        if self.defaulted or self.prepaid:
            return 0.0

        if self.externally_defaulted:
            # Treat as immediate default with recovery; log recovery into payment_record
            self.defaulted = True
            recovery_rate = 0.30
            recovery_amount = float(self.remaining_balance) * recovery_rate
            self.remaining_balance = 0.0
            if recovery_amount > 0.0:
                self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
            return recovery_amount

        # Add this week’s scheduled installment
        if week in self.expected_payment_schedule:
            self.due_stack.append((week, self.installment_amount, 0.0))

        total_payment = 0.0
        still_due = []

        # Credit-score-driven parameters
        if self.credit_score < 600:
            miss_payment_chance = 0.12
            default_threshold = 4
        elif self.credit_score < 700:
            miss_payment_chance = 0.07
            default_threshold = 5
        else:
            miss_payment_chance = 0.02
            default_threshold = 6

        # Process all outstanding dues
        for due_week, amount_due, late_fee in self.due_stack:

            # Accumulate late fees if overdue by >= 2 weeks
            if (week - due_week) >= 2:
                extra_fee = min(
                    self.late_fee,
                    0.25 * self.order_amount - self.total_late_fees_paid
                )
                late_fee += max(0.0, extra_fee)

            total_due = amount_due + late_fee

            # Random chance of missing payment
            will_miss_payment = random.random() < miss_payment_chance

            if will_miss_payment or self.remaining_balance < total_due:
                # Payment missed → carry forward
                still_due.append((due_week, amount_due, late_fee))

                # Count as a missed obligation every week until paid
                self.missed_payment_count += 1
                if week not in self.late_weeks:
                    self.late_weeks.append(week)

                # Trigger default if threshold reached
                if self.missed_payment_count >= default_threshold:
                    self.defaulted = True
                    recovery_rate = 0.30
                    recovery_amount = self.remaining_balance * recovery_rate
                    self.remaining_balance = 0.0
                    if recovery_amount > 0:
                        self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
                    return total_payment + recovery_amount

            else:
                # Borrower pays full obligation
                self.remaining_balance -= total_due
                total_payment += total_due
                self.total_late_fees_paid += late_fee
                self.paid_weeks.append(week)
                self.payment_record[week] = self.payment_record.get(week, 0.0) + total_due

                if week != due_week:
                    if not hasattr(self, "payment_delays"):
                        self.payment_delays = []
                    self.payment_delays.append(week - due_week)

        # Replace with still-due items
        self.due_stack = still_due

        # Prepayment chance only if not defaulted and balance > 0
        if (not self.prepaid 
            and week in self.expected_payment_schedule 
            and random.random() < 0.05):
            prepay_amount = self.remaining_balance
            self.remaining_balance = 0.0
            self.prepaid = True
            total_payment += prepay_amount
            self.payment_record[week] = self.payment_record.get(week, 0.0) + prepay_amount

        # Mark fully prepaid if balance cleared
        if self.remaining_balance <= 1e-8:
            self.prepaid = True

        return total_payment if not self.defaulted else 0.0



class ValuationLoan:
    def __init__(self, loan_id, order_amount, credit_score):
        self.loan_id = int(loan_id)
        self.order_amount = float(order_amount)
        self.credit_score = int(credit_score)
        self.payment_record = {}
        self.paid_weeks = []
        self.late_weeks = []
        self.total_late_fees_paid = 0.0

        # State
        self.remaining_balance = float(order_amount)
        self.defaulted = False
        self.prepaid = False

        # Hybrid copula scheduling
        self.default_time_week_continuous = float("inf")
        self.default_week = None
        self.default_scheduled_week_to_pay = None  # next due week >= default time

        # Payment schedule
        self.payment_weeks = (2, 4, 6, 8)
        self.installment = self.order_amount / 4.0

        # Optional accounting
        self.late_fees_collected = 0.0
        self.cash_collected = 0.0

        # For richer, path-like payment realism
        if not hasattr(self, "expected_payment_schedule"):
            self.expected_payment_schedule = [2, 4, 6, 8]
        if not hasattr(self, "due_stack"):
            self.due_stack = []  # list of tuples: (due_week, amount_due, late_fee)
        if not hasattr(self, "late_fee"):
            self.late_fee = 5  # per-late-fee increment (capped overall)
        if not hasattr(self, "total_late_fees_paid"):
            self.total_late_fees_paid = 0.0
        if not hasattr(self, "missed_payment_count"):
            self.missed_payment_count = 0
        if not hasattr(self, "installment_amount"):
            self.installment_amount = self.installment
        if not hasattr(self, "externally_defaulted"):
            self.externally_defaulted = False

    def _trigger_default_if_due(self, week: int) -> None:
        if self.defaulted:
            return
        if self.default_scheduled_week_to_pay is not None:
            if week >= self.default_scheduled_week_to_pay:
                self.defaulted = True

                # Add simple recovery on default
                recovery_rate = 0.30
                recovery_amount = self.remaining_balance * recovery_rate
                self.remaining_balance = 0.0
                if recovery_amount > 0:
                    self.cash_collected += recovery_amount
                    self.payment_record[week] = (
                        self.payment_record.get(week, 0.0) + recovery_amount
                    )

    def simulate_payment(self, week: int) -> float:
        # Idempotency: avoid double-posting for the same week
        if week in self.payment_record:
            return 0.0

        # If already settled states
        if self.defaulted or self.prepaid:
            return 0.0

        # Empirical default time override
        if hasattr(self, "sampled_default_time") and not self.defaulted:
            sampled_floor = int(np.floor(self.sampled_default_time))
            if week >= sampled_floor:
                self.defaulted = True
                self.default_week = sampled_floor
                self.default_time_week_continuous = self.sampled_default_time
                self.default_scheduled_week_to_pay = week
                # Recovery on default
                recovery_rate = 0.30
                recovery_amount = float(self.remaining_balance) * recovery_rate
                self.remaining_balance = 0.0
                if recovery_amount > 0.0:
                    self.cash_collected += recovery_amount
                    self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
                return recovery_amount

        # External/default override hook (kept for parity)
        if getattr(self, "externally_defaulted", False):
            # Treat as immediate default with recovery; log into payment_record
            self.defaulted = True
            recovery_rate = 0.30
            recovery_amount = float(self.remaining_balance) * recovery_rate
            self.remaining_balance = 0.0
            if recovery_amount > 0.0:
                self.cash_collected += recovery_amount
                self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
            return recovery_amount

        # --- Copula-scheduled default takes precedence ---
        if self.default_scheduled_week_to_pay is not None and week >= self.default_scheduled_week_to_pay:
            self.defaulted = True
            # Recovery on default (simple, immediate)
            recovery_rate = 0.30
            recovery_amount = float(self.remaining_balance) * recovery_rate
            self.remaining_balance = 0.0
            if recovery_amount > 0.0:
                self.cash_collected += recovery_amount
                self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
            return recovery_amount

        # Empirical payment delay override
        if hasattr(self, "sampled_delay") and self.sampled_delay > 0:
            self.sampled_delay -= 1
            return 0.0

        # Add this week’s scheduled installment obligation
        if week in self.payment_weeks:
            self.due_stack.append((week, float(self.installment_amount), 0.0))

        total_payment = 0.0
        still_due = []

        # Credit-score-driven parameters (same buckets as Loan)
        if self.credit_score < 600:
            miss_payment_chance = 0.12
            default_threshold = 4
        elif self.credit_score < 700:
            miss_payment_chance = 0.07
            default_threshold = 5
        else:
            miss_payment_chance = 0.02
            default_threshold = 6

        # Process all outstanding dues
        for due_week, amount_due, late_fee in self.due_stack:

            # Accumulate late fees if overdue by >= 2 weeks (cap overall at 25% of order amount)
            if (week - due_week) >= 2:
                extra_fee_cap = max(0.0, 0.25 * self.order_amount - float(self.total_late_fees_paid))
                extra_fee = min(float(self.late_fee), extra_fee_cap)
                late_fee += extra_fee

            total_due = amount_due + late_fee

            # Random chance of missing this obligation
            will_miss_payment = random.random() < miss_payment_chance

            if will_miss_payment or self.remaining_balance < total_due:
                # Carry forward unpaid obligation
                still_due.append((due_week, amount_due, late_fee))

                # Count as a missed obligation every week until paid
                self.missed_payment_count += 1
                if week not in self.late_weeks:
                    self.late_weeks.append(week)

                # Trigger behavioral default if threshold reached (only if not already forced by copula)
                if self.missed_payment_count >= default_threshold:
                    self.defaulted = True
                    recovery_rate = 0.30
                    recovery_amount = float(self.remaining_balance) * recovery_rate
                    self.remaining_balance = 0.0
                    if recovery_amount > 0.0:
                        self.cash_collected += recovery_amount
                        self.payment_record[week] = self.payment_record.get(week, 0.0) + recovery_amount
                        
                    return total_payment + recovery_amount

            else:
                # Pay this obligation in full
                self.remaining_balance -= total_due
                total_payment += total_due
                self.total_late_fees_paid += late_fee
                if week not in self.paid_weeks:
                    self.paid_weeks.append(week)
                self.payment_record[week] = self.payment_record.get(week, 0.0) + total_due

                if week != due_week:
                    if not hasattr(self, "payment_delays"):
                        self.payment_delays = []
                    self.payment_delays.append(week - due_week)

        # Replace with still-due items for next weeks
        self.due_stack = still_due

        # Optional prepayment chance (after obligations processed)
        if (not self.prepaid) and (week in self.payment_weeks) and (random.random() < 0.05):
            prepay_amount = float(self.remaining_balance)
            self.remaining_balance = 0.0
            self.prepaid = True
            total_payment += prepay_amount
            self.payment_record[week] = self.payment_record.get(week, 0.0) + prepay_amount

        # Mark fully prepaid if balance cleared
        if self.remaining_balance <= 1e-8:
            self.prepaid = True

        return total_payment if not self.defaulted else 0.0




def generate_correlated_defaults(n_loans, default_probs, rho=0.2, seed=None):
    """
    Generate correlated default flags using a Gaussian copula.

    Parameters:
        n_loans (int): Number of loans
        default_probs (list or np.array): Marginal default probabilities for each loan
        rho (float): Correlation coefficient (between 0 and 1)
        seed (int or None): Random seed for reproducibility

    Returns:
        np.array of bools: True means the loan defaulted
    """
    if seed is not None:
        np.random.seed(seed)

    # Create correlation matrix
    corr_matrix = rho * np.ones((n_loans, n_loans)) + (1 - rho) * np.eye(n_loans)
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent standard normals
    Z_indep = np.random.normal(size=(n_loans,))
    Z_corr = L @ Z_indep

    # Determine default based on inverse CDF (quantile function)
    thresholds = norm.ppf(default_probs)
    return Z_corr < thresholds