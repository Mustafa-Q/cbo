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