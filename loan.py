from copula import score_to_default_rate
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

    def simulate_payment(self, week):
        if self.defaulted or self.prepaid:
            return 0

        if self.externally_defaulted:
            self.set_default()
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



class ValuationLoan:
    def __init__(self, loan_id, order_amount, credit_score, externally_defaulted=None):
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
        self.externally_defaulted = externally_defaulted

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

    def set_default(self):
        self.defaulted = True
        self.remaining_balance = 0

    def simulate_payment(self, week):
        if self.defaulted or self.prepaid or self.remaining_balance <= 0:
            return 0

        if self.externally_defaulted:
            self.set_default()
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