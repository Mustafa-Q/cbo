# Add helper imports after any existing imports (none in this file)
from copula import generate_correlated_defaults, score_to_default_rate

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
# Assign Correlated Defaults Function
# ----------------------------
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

