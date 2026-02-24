# Dynamic Alpha Quantile Policy - Execution Flow

## 📋 Implementation Overview

This document outlines the complete execution flow for the Dynamic Alpha Quantile Policy implementation, including verification steps and testing procedures.

---

## 🎯 Phase 1: Initial Verification (Now)

### Step 1.1: Syntax and Structure Check
```bash
cd "e:\研究活\你一定要顺利毕业啊\FICA-main"
python verify_implementation.py
```

**Expected Output:**
```
======================================================================
QUANTILE POLICY - IMPLEMENTATION VERIFICATION
======================================================================

Checking: quantile_policy.py
======================================================================
  ✓ File exists
  ✓ Syntax is valid
  ✓ All required imports present
  ✓ Class QuantilePolicy found

[Similar checks for other files...]

======================================================================
✓ ALL CHECKS PASSED!
======================================================================
```

### Step 1.2: Git Status Check
```bash
git status
```

**Expected Output:**
- 7 new files added
- Committed with hash `86170b1`
- Pushed to `origin/myexp`

---

## 🧪 Phase 2: Automated Testing (Next)

### Step 2.1: Activate FICA Environment

**Windows (PowerShell):**
```powershell
cd "e:\研究活\你一定要顺利毕业啊\FICA-main"
.\activate_env.bat
```

**Or manually:**
```bash
conda activate fica
```

### Step 2.2: Run Full Test Suite
```bash
python test_quantile.py
```

**Test Breakdown:**

| Test | Description | Expected Time | Status |
|------|-------------|---------------|--------|
| TEST 1 | Basic functionality | ~5 seconds | ⏳ Pending |
| TEST 2 | Scenario evaluation | ~5 seconds | ⏳ Pending |
| TEST 3 | Save/Load | ~3 seconds | ⏳ Pending |
| TEST 4 | Full evaluation (small) | ~1-2 minutes | ⏳ Pending |

**Expected Output:**
```
======================================================================
QUANTILE POLICY - AUTOMATED TESTING
======================================================================

======================================================================
TEST 1: Basic Functionality
======================================================================
[...]

✓ TEST 1 PASSED: Basic functionality works correctly

======================================================================
TEST 2: Scenario Evaluation
======================================================================
[...]
  quantile    : cost = $12,345.67, feasible = True
  FICA        : cost = $12,456.78, feasible = True
  worst       : cost = $13,567.89, feasible = True
  best        : cost = $11,234.56, feasible = True

✓ TEST 2 PASSED: Scenario evaluation works correctly

[...]

======================================================================
TEST SUMMARY
======================================================================
Tests passed: 4/4
✓ ALL TESTS PASSED!
======================================================================
```

---

## 📊 Phase 3: Full Evaluation (After Tests Pass)

### Step 3.1: Run Medium-Scale Evaluation
```bash
python evaluate_quantile.py
```

Or with custom parameters:
```python
from evaluate_quantile import run_full_evaluation

evaluator, results = run_full_evaluation(
    num_gen=2,
    num_WT=2,
    T=24,
    N_WDR=100,
    epsilon=0.05,
    theta=0.15,
    norm_ord=1,
    N_test=100
)
```

**Expected Timeline:**
- Data setup: ~10 seconds
- FICA optimization: ~2-3 minutes
- Quantile policy initialization: ~5 seconds
- Scenario evaluation: ~1-2 minutes
- Total: ~5-8 minutes

### Step 3.2: Verify Output Files

Check that these files are created in `figure/quantile_evaluation/`:

```
figure/quantile_evaluation/
├── eval_results_gen2_WT2_T24.pkl
├── policy_gen2_WT2_T24.pkl
├── cost_distribution.png
├── lambda_distribution.png
├── cost_improvement.png
└── summary_table.png
```

### Step 3.3: Review Results

**Key Metrics to Check:**

1. **Cost Comparison:**
   - Quantile Policy mean cost: Should be close to or lower than FICA
   - Improvement rate: 2-10% expected
   - Win rate: 60-80% expected

2. **Lambda Distribution:**
   - Should vary across time (not constant)
   - Should vary across scenarios
   - Range should be within [0, 1]

3. **Feasibility:**
   - Both strategies should have high feasibility (>95%)

---

## 🔬 Phase 4: Advanced Experiments (Optional)

### Experiment 4.1: Parameter Sensitivity Analysis

**Vary Risk Level (ε):**
```python
epsilon_values = [0.01, 0.03, 0.05, 0.07, 0.10]
for eps in epsilon_values:
    evaluator, results = run_full_evaluation(epsilon=eps, N_test=50)
```

**Vary Wasserstein Radius (θ):**
```python
theta_values = [0.10, 0.13, 0.15, 0.17, 0.20]
for theta in theta_values:
    evaluator, results = run_full_evaluation(theta=theta, N_test=50)
```

**Vary System Size:**
```python
system_configs = [
    (2, 2),  # num_gen=2, num_WT=2
    (3, 2),
    (5, 3),
    (10, 5),
]
for num_gen, num_WT in system_configs:
    evaluator, results = run_full_evaluation(
        num_gen=num_gen, num_WT=num_WT, N_test=50
    )
```

### Experiment 4.2: Large-Scale Test
```python
evaluator, results = run_full_evaluation(
    num_gen=5,
    num_WT=3,
    T=24,
    N_WDR=200,
    N_test=500
)
```

**Expected Timeline:** 30-60 minutes

---

## 📈 Phase 5: Result Analysis

### Step 5.1: Visual Inspection

Review generated plots:

1. **Cost Distribution:**
   - Compare histograms of quantile vs FICA
   - Check if quantile has lower mean cost
   - Identify any pathological cases

2. **Lambda Distribution:**
   - Analyze temporal patterns
   - Check correlation with load patterns
   - Identify peak/off-peak differences

3. **Cost Improvement:**
   - Check if improvement is skewed or symmetric
   - Identify scenarios where quantile performs worse

### Step 5.2: Statistical Analysis

Calculate additional metrics:

```python
import numpy as np
import scipy.stats as stats

# Load results
results = joblib.load('figure/quantile_evaluation/eval_results_gen2_WT2_T24.pkl')

# Statistical tests
costs_quantile = results['results']['costs_quantile']
costs_FICA = results['results']['costs_FICA']

# Paired t-test
t_stat, p_value = stats.ttest_rel(costs_FICA, costs_quantile)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

# Wilcoxon signed-rank test (non-parametric)
w_stat, p_value_w = stats.wilcoxon(costs_FICA, costs_quantile)
print(f"Wilcoxon test: W={w_stat:.4f}, p={p_value_w:.4f}")

# Effect size (Cohen's d)
cohen_d = np.mean(costs_FICA - costs_quantile) / np.std(costs_FICA - costs_quantile)
print(f"Cohen's d: {cohen_d:.4f}")
```

### Step 5.3: Generate Report

Create a comprehensive report including:

1. **Executive Summary**
   - Key findings
   - Performance metrics
   - Recommendations

2. **Methodology**
   - System configuration
   - Parameter settings
   - Evaluation procedure

3. **Results**
   - Cost comparison tables
   - Statistical analysis
   - Visualizations

4. **Discussion**
   - Interpretation of results
   - Comparison with expectations
   - Limitations

5. **Conclusions**
   - Main takeaways
   - Future work

---

## 🐛 Troubleshooting Guide

### Issue 1: Import Errors
**Symptom:** `ModuleNotFoundError: No module named 'XXX'`

**Solution:**
```bash
# Ensure FICA environment is activated
conda activate fica

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Issue 2: Memory Errors
**Symptom:** `MemoryError` or system becomes unresponsive

**Solution:**
```python
# Reduce problem size
config = QuantileConfig.small_test()  # Use smaller configuration
```

### Issue 3: Solver Timeouts
**Symptom:** Gurobi solver exceeds time limit

**Solution:**
```python
# In quantile_config.py, modify:
MIPGap: float = 0.005  # Relax optimality tolerance
thread: int = 8  # Use more threads (if available)
```

### Issue 4: Infeasibility
**Symptom:** Problem becomes infeasible

**Solution:**
```python
# Check and adjust load scaling
load_scaling_factor: float = 0.8  # Reduce load
```

---

## ✅ Success Criteria

The implementation is considered successful when:

1. **All Tests Pass:**
   - [ ] TEST 1: Basic functionality ✓
   - [ ] TEST 2: Scenario evaluation ✓
   - [ ] TEST 3: Save/Load ✓
   - [ ] TEST 4: Full evaluation ✓

2. **Evaluation Completes:**
   - [ ] FICA optimization completes successfully
   - [ ] All test scenarios evaluated
   - [ ] Results saved correctly

3. **Performance Meets Expectations:**
   - [ ] Quantile policy cost ≤ FICA cost (on average)
   - [ ] Win rate ≥ 60%
   - [ ] Feasibility rate ≥ 95%
   - [ ] Lambda distribution shows variation

4. **Documentation Complete:**
   - [ ] README written
   - [ ] API documented
   - [ ] Examples provided

---

## 📝 Next Steps

After successful implementation and testing:

1. **Paper Integration:**
   - Add results to manuscript
   - Include mathematical formulation
   - Provide case studies

2. **Extended Experiments:**
   - Test on larger systems
   - Compare with other baselines
   - Perform sensitivity analysis

3. **Code Optimization:**
   - Profile performance
   - Optimize bottlenecks
   - Add parallelization

4. **Documentation:**
   - Create tutorial notebooks
   - Add inline documentation
   - Improve examples

---

## 📞 Support

If you encounter issues:

1. Check troubleshooting section above
2. Review `QUANTILE_POLICY_README.md`
3. Examine test outputs for error messages
4. Verify environment configuration

---

**Last Updated:** 2025-02-24
**Version:** 1.0.0
**Status:** Implementation Complete, Testing Pending
