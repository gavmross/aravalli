"""Tests for src/amortization.py"""

import numpy as np
import pandas as pd
import pytest

from src.amortization import calc_monthly_payment, calc_balance, calc_payment_num, calc_amort


class TestCalcMonthlyPayment:
    """Test the monthly payment calculation."""

    def test_known_loan(self):
        """$10,000 at 10% for 36 months = $322.67/month."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.10])
        term = np.array([36])
        payment = calc_monthly_payment(principal, annual_rate, term)
        assert round(payment[0], 2) == 322.67

    def test_zero_interest_rate(self):
        """Zero rate: payment = principal / term."""
        principal = np.array([12000.0])
        annual_rate = np.array([0.0])
        term = np.array([36])
        payment = calc_monthly_payment(principal, annual_rate, term)
        assert round(payment[0], 2) == 333.33

    def test_vectorized(self):
        """Multiple loans at once."""
        principal = np.array([10000.0, 20000.0])
        annual_rate = np.array([0.10, 0.08])
        term = np.array([36, 60])
        payments = calc_monthly_payment(principal, annual_rate, term)
        assert len(payments) == 2
        assert round(payments[0], 2) == 322.67
        assert round(payments[1], 2) == 405.53

    def test_high_rate(self):
        """High interest rate (30%) should still compute."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.30])
        term = np.array([36])
        payment = calc_monthly_payment(principal, annual_rate, term)
        assert payment[0] > 0
        # At 30%, payment should be higher than at 10%
        assert payment[0] > 322.67


class TestCalcBalance:
    """Test remaining balance calculation."""

    def test_full_term(self):
        """After all payments, balance should be ~0."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.10])
        term = np.array([36])
        payment = calc_monthly_payment(principal, annual_rate, term)
        balance, prin_paid, int_paid = calc_balance(principal, annual_rate, payment, term)
        assert abs(balance[0]) < 0.01  # Should be ~0
        assert abs(prin_paid[0] - 10000.0) < 0.01  # All principal repaid

    def test_zero_payments(self):
        """No payments made: balance equals principal."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.10])
        payment = np.array([322.67])
        payments_made = np.array([0])
        balance, prin_paid, int_paid = calc_balance(principal, annual_rate, payment, payments_made)
        assert balance[0] == 10000.0
        assert prin_paid[0] == 0.0

    def test_partial_payments(self):
        """After some payments, balance should be between 0 and principal."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.10])
        term = np.array([36])
        payment = calc_monthly_payment(principal, annual_rate, term)
        payments_made = np.array([18])  # Halfway through
        balance, _, _ = calc_balance(principal, annual_rate, payment, payments_made)
        assert 0 < balance[0] < 10000.0

    def test_balance_decreases_monotonically(self):
        """Balance should decrease with each payment."""
        principal = np.array([10000.0])
        annual_rate = np.array([0.10])
        payment = calc_monthly_payment(principal, annual_rate, np.array([36]))
        prev_balance = 10000.0
        for n in range(1, 37):
            bal, _, _ = calc_balance(principal, annual_rate, payment, np.array([n]))
            assert bal[0] < prev_balance
            prev_balance = bal[0]


class TestCalcPaymentNum:
    """Test month difference calculation."""

    def test_same_month(self):
        """Same month should return 0."""
        start = pd.Series([pd.Timestamp('2019-01-01')])
        end = pd.Series([pd.Timestamp('2019-01-01')])
        result = calc_payment_num(start, end)
        assert result[0] == 0

    def test_one_year(self):
        """12 months apart."""
        start = pd.Series([pd.Timestamp('2018-01-01')])
        end = pd.Series([pd.Timestamp('2019-01-01')])
        result = calc_payment_num(start, end)
        assert result[0] == 12

    def test_36_months(self):
        """36 months for a typical loan."""
        start = pd.Series([pd.Timestamp('2016-03-01')])
        end = pd.Series([pd.Timestamp('2019-03-01')])
        result = calc_payment_num(start, end)
        assert result[0] == 36

    def test_negative_clamped_to_zero(self):
        """End before start should return 0."""
        start = pd.Series([pd.Timestamp('2019-06-01')])
        end = pd.Series([pd.Timestamp('2019-01-01')])
        result = calc_payment_num(start, end)
        assert result[0] == 0


class TestCalcAmort:
    """Test the full amortization function."""

    @pytest.fixture
    def sample_df(self):
        """Create a minimal DataFrame for testing calc_amort."""
        return pd.DataFrame({
            'funded_amnt': [10000.0],
            'int_rate': [0.10],
            'term_months': [36],
            'issue_d': [pd.Timestamp('2016-03-01')],
            'last_pymnt_d': [pd.Timestamp('2019-03-01')],
            'out_prncp': [0.0],
            'last_pymnt_amnt': [320.0],
            'installment': [322.67],
        })

    def test_columns_created(self, sample_df):
        """calc_amort should add expected columns."""
        result = calc_amort(sample_df)
        expected_cols = [
            'orig_exp_monthly_payment', 'orig_exp_payments_made',
            'orig_exp_balance', 'orig_exp_principal_paid', 'orig_exp_interest_paid',
            'orig_exp_monthly_principal', 'orig_exp_monthly_interest',
            'orig_exp_balance_diff', 'orig_exp_installment_diff',
            'last_pmt_beginning_balance', 'last_pmt_interest',
            'last_pmt_actual_principal', 'last_pmt_scheduled_principal',
            'last_pmt_unscheduled_principal', 'last_pmt_smm', 'last_pmt_cpr',
            'next_pmt_principal', 'next_pmt_interest',
            'updated_remaining_term', 'updated_maturity_date',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_monthly_payment_matches(self, sample_df):
        """Expected monthly payment should match known value."""
        result = calc_amort(sample_df)
        assert result['orig_exp_monthly_payment'].iloc[0] == 322.67

    def test_fully_paid_balance_near_zero(self, sample_df):
        """Loan with 36 payments on a 36-month term should have ~0 expected balance."""
        result = calc_amort(sample_df)
        assert abs(result['orig_exp_balance'].iloc[0]) < 1.0

    def test_does_not_modify_input(self, sample_df):
        """calc_amort should not modify the input DataFrame."""
        original_cols = list(sample_df.columns)
        _ = calc_amort(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_missing_column_raises(self):
        """Missing required column should raise ValueError."""
        df = pd.DataFrame({'funded_amnt': [10000.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calc_amort(df)

    def test_percentage_rate_auto_converts(self):
        """Rate > 1 should be auto-converted from percentage to decimal."""
        df = pd.DataFrame({
            'funded_amnt': [10000.0],
            'int_rate': [10.0],  # percentage form
            'term_months': [36],
            'issue_d': [pd.Timestamp('2016-03-01')],
            'last_pymnt_d': [pd.Timestamp('2019-03-01')],
            'out_prncp': [0.0],
            'last_pymnt_amnt': [320.0],
            'installment': [322.67],
        })
        result = calc_amort(df)
        # Should still compute same payment as decimal form
        assert result['orig_exp_monthly_payment'].iloc[0] == 322.67

    def test_string_dates_auto_converted(self):
        """String dates should be parsed automatically."""
        df = pd.DataFrame({
            'funded_amnt': [10000.0],
            'int_rate': [0.10],
            'term_months': [36],
            'issue_d': ['2016-03-01'],
            'last_pymnt_d': ['2019-03-01'],
            'out_prncp': [0.0],
            'last_pymnt_amnt': [320.0],
            'installment': [322.67],
        })
        result = calc_amort(df)
        assert result['orig_exp_monthly_payment'].iloc[0] == 322.67
