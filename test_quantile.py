"""
Quick Test Script for Quantile Policy Implementation
======================================================

This script provides a quick test to verify the quantile policy implementation.
"""

import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from quantile_policy import QuantilePolicy
from evaluate_quantile import run_full_evaluation


def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    # Create synthetic data
    T = 10
    num_gen = 3
    num_WT = 2
    N_samples = 50

    # Generate synthetic data
    gen_power_all = np.random.uniform(50, 100, (T, num_gen))
    gen_alpha_all = np.random.uniform(-1, 1, (T, num_gen))
    WT_error_scenarios = np.random.randn(N_samples, T, num_WT) * 10
    gen_cap_individual = np.array([120] * num_gen)
    gen_pmin_individual = np.array([20] * num_gen)
    gen_cost = np.random.uniform(30, 50, num_gen)

    # Create policy
    policy = QuantilePolicy(T=T, num_gen=num_gen, num_WT=num_WT)

    # Solve offline
    offline_results = policy.solve_offline(
        gen_power_all, gen_alpha_all, WT_error_scenarios,
        gen_cap_individual, gen_pmin_individual, gen_cost
    )

    # Test online dispatch
    WT_pred = np.random.uniform(40, 60, (T, num_WT))
    WT_actual = WT_pred + np.random.randn(T, num_WT) * 5

    x_real, lambda_t = policy.online_dispatch(WT_pred, WT_actual, verbose=True)

    # Verify shape
    assert x_real.shape == (T, num_gen), f"Expected shape (T, num_gen), got {x_real.shape}"
    assert lambda_t.shape == (T,), f"Expected shape (T,), got {lambda_t.shape}"
    assert np.all(lambda_t >= 0) and np.all(lambda_t <= 1), "Lambda should be in [0, 1]"

    print("\n✓ TEST 1 PASSED: Basic functionality works correctly")
    return True


def test_scenario_evaluation():
    """Test scenario evaluation."""
    print("\n" + "=" * 70)
    print("TEST 2: Scenario Evaluation")
    print("=" * 70)

    T = 10
    num_gen = 3
    num_WT = 2
    N_samples = 50

    # Generate synthetic data
    gen_power_all = np.random.uniform(50, 100, (T, num_gen))
    gen_alpha_all = np.random.uniform(-1, 1, (T, num_gen))
    WT_error_scenarios = np.random.randn(N_samples, T, num_WT) * 10
    gen_cap_individual = np.array([120] * num_gen)
    gen_pmin_individual = np.array([20] * num_gen)
    gen_cost = np.random.uniform(30, 50, num_gen)
    gen_cost_quadra = np.random.uniform(0.001, 0.01, num_gen)

    # Create and initialize policy
    policy = QuantilePolicy(T=T, num_gen=num_gen, num_WT=num_WT)
    policy.solve_offline(
        gen_power_all, gen_alpha_all, WT_error_scenarios,
        gen_cap_individual, gen_pmin_individual, gen_cost
    )

    # Test scenario evaluation
    WT_pred = np.random.uniform(40, 60, (T, num_WT))
    WT_error_scenario = np.random.randn(T, num_WT) * 5

    for strategy in ['quantile', 'FICA', 'worst', 'best']:
        result = policy.evaluate_scenario(
            WT_pred, WT_error_scenario, gen_cost_quadra, strategy=strategy
        )
        print(f"  {strategy:12s}: cost = ${result['cost']:,.2f}, feasible = {result['feasible']}")

    print("\n✓ TEST 2 PASSED: Scenario evaluation works correctly")
    return True


def test_save_load():
    """Test save and load functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: Save and Load")
    print("=" * 70)

    T = 10
    num_gen = 3
    num_WT = 2
    N_samples = 50

    # Generate synthetic data
    gen_power_all = np.random.uniform(50, 100, (T, num_gen))
    gen_alpha_all = np.random.uniform(-1, 1, (T, num_gen))
    WT_error_scenarios = np.random.randn(N_samples, T, num_WT) * 10
    gen_cap_individual = np.array([120] * num_gen)
    gen_pmin_individual = np.array([20] * num_gen)
    gen_cost = np.random.uniform(30, 50, num_gen)

    # Create and save policy
    policy1 = QuantilePolicy(T=T, num_gen=num_gen, num_WT=num_WT)
    policy1.solve_offline(
        gen_power_all, gen_alpha_all, WT_error_scenarios,
        gen_cap_individual, gen_pmin_individual, gen_cost
    )

    # Save
    test_file = 'test_policy_temp.pkl'
    policy1.save(test_file)

    # Load
    policy2 = QuantilePolicy.load(test_file)

    # Verify
    assert np.allclose(policy1.gen_power_all, policy2.gen_power_all), "gen_power_all mismatch"
    assert np.allclose(policy1.gen_alpha_all, policy2.gen_alpha_all), "gen_alpha_all mismatch"
    assert np.allclose(policy1.x_worst, policy2.x_worst), "x_worst mismatch"
    assert np.allclose(policy1.x_best, policy2.x_best), "x_best mismatch"

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n✓ TEST 3 PASSED: Save and load works correctly")
    return True


def test_full_evaluation_small():
    """Test full evaluation with small problem size."""
    print("\n" + "=" * 70)
    print("TEST 4: Full Evaluation (Small Scale)")
    print("=" * 70)
    print("Note: This test may take 1-2 minutes...")

    try:
        evaluator, results = run_full_evaluation(
            num_gen=2,
            num_WT=2,
            T=12,  # Reduced from 24
            N_WDR=50,  # Reduced from 100
            epsilon=0.05,
            theta=0.15,
            norm_ord=1,
            N_test=20  # Reduced from 100
        )

        print("\n✓ TEST 4 PASSED: Full evaluation works correctly")
        return True
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("QUANTILE POLICY - AUTOMATED TESTING")
    print("=" * 70)

    tests_passed = 0
    tests_total = 4

    # Run tests
    if test_basic_functionality():
        tests_passed += 1

    if test_scenario_evaluation():
        tests_passed += 1

    if test_save_load():
        tests_passed += 1

    if test_full_evaluation_small():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    if tests_passed == tests_total:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {tests_total - tests_passed} test(s) failed")
    print("=" * 70)

    return tests_passed == tests_total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
