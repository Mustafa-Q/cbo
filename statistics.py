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