"""
Dynamic Alpha Quantile Policy for FICA
======================================

This module implements the dynamic quantile-based dispatch policy that improves
upon the standard FICA approach by utilizing real-time observation of wind power
output to interpolate between worst-case and best-case strategies.

Mathematical Formulation:
    x_t(λ) = λ_t * ẋ_t + (1-λ_t) * x̂_t

where:
    - ẋ_t: worst-case strategy (maximum error scenario)
    - x̂_t: best-case strategy (minimum error scenario)
    - λ_t: quantile parameter based on observed error sum

Key Features:
    - Offline optimization phase computes base FICA solution and extreme strategies
    - Online dispatch phase uses real-time observations for adaptive dispatch
    - Maintains mathematical consistency with original FICA framework
"""

import numpy as np
import os
import time
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed

plt.style.use('default')
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    "mathtext.fontset": "cm",
})


class QuantilePolicy:
    """
    Dynamic Alpha Quantile Policy for renewable energy dispatch.

    This policy improves upon standard FICA by utilizing real-time wind power
    observations to adaptively interpolate between extreme strategies.
    """

    def __init__(self, T: int, num_gen: int, num_WT: int,
                 epsilon: float = 0.05, theta: float = 0.15):
        """
        Initialize the Quantile Policy.

        Parameters:
        -----------
        T : int
            Time horizon (number of time steps)
        num_gen : int
            Number of thermal generators
        num_WT : int
            Number of wind farms
        epsilon : float
            Risk level for joint chance constraints
        theta : float
            Wasserstein radius
        """
        self.T = T
        self.num_gen = num_gen
        self.num_WT = num_WT
        self.epsilon = epsilon
        self.theta = theta

        # Storage for offline optimization results
        self.gen_power_all = None  # First-stage power (T, num_gen)
        self.gen_alpha_all = None   # AGC factors (T, num_gen)
        self.x_worst = None         # Worst-case strategy (T, num_gen)
        self.x_best = None          # Best-case strategy (T, num_gen)
        self.error_worst_t = None   # Worst error sum for each t (T,)
        self.error_best_t = None    # Best error sum for each t (T,)

        # Additional metadata
        self.gen_cap_individual = None
        self.gen_pmin_individual = None
        self.gen_cost = None

    def solve_offline(self, gen_power_all: np.ndarray, gen_alpha_all: np.ndarray,
                      WT_error_scenarios_train: np.ndarray,
                      gen_cap_individual: np.ndarray,
                      gen_pmin_individual: np.ndarray,
                      gen_cost: np.ndarray) -> Dict:
        """
        Offline optimization phase: compute extreme strategies from base FICA solution.

        Parameters:
        -----------
        gen_power_all : np.ndarray
            First-stage power output from FICA (T, num_gen)
        gen_alpha_all : np.ndarray
            AGC factors from FICA (T, num_gen)
        WT_error_scenarios_train : np.ndarray
            Training error scenarios (N_samples, T, num_WT)
        gen_cap_individual : np.ndarray
            Generator capacity limits (num_gen,)
        gen_pmin_individual : np.ndarray
            Generator minimum output limits (num_gen,)
        gen_cost : np.ndarray
            Generator cost coefficients (num_gen,)

        Returns:
        --------
        results : Dict
            Dictionary containing offline optimization results
        """
        print("=" * 60)
        print("Starting Offline Optimization Phase...")
        print("=" * 60)

        # Store base solution
        self.gen_power_all = gen_power_all.copy()
        self.gen_alpha_all = gen_alpha_all.copy()
        self.gen_cap_individual = gen_cap_individual.copy()
        self.gen_pmin_individual = gen_pmin_individual.copy()
        self.gen_cost = gen_cost.copy()

        # Calculate error sums for each scenario at each time step
        # Shape: (N_samples, T)
        error_sums = WT_error_scenarios_train.sum(axis=-1)

        # Find worst and best indices for each time step
        # Worst: minimum error sum (most negative wind forecast error)
        self.error_worst_t = np.min(error_sums, axis=0)  # (T,)
        self.error_best_t = np.max(error_sums, axis=0)  # (T,)

        # Find corresponding scenario indices
        worst_indices = np.argmin(error_sums, axis=0)  # (T,)
        best_indices = np.argmax(error_sums, axis=0)   # (T,)

        # Compute extreme strategies
        # Worst case: ẋ_t = P_g^0 - α_g * e_worst_t
        self.x_worst = np.zeros((self.T, self.num_gen))
        for t in range(self.T):
            self.x_worst[t, :] = (self.gen_power_all[t, :] -
                                  self.gen_alpha_all[t, :] * self.error_worst_t[t])

        # Best case: x̂_t = P_g^0 - α_g * e_best_t
        self.x_best = np.zeros((self.T, self.num_gen))
        for t in range(self.T):
            self.x_best[t, :] = (self.gen_power_all[t, :] -
                                 self.gen_alpha_all[t, :] * self.error_best_t[t])

        # Validate extreme strategies
        self._validate_extreme_strategies()

        print(f"✓ Offline optimization completed successfully")
        print(f"  - Worst error range: [{self.error_worst_t.min():.2f}, {self.error_worst_t.max():.2f}] MW")
        print(f"  - Best error range: [{self.error_best_t.min():.2f}, {self.error_best_t.max():.2f}] MW")
        print(f"  - Worst strategy range: [{self.x_worst.min():.2f}, {self.x_worst.max():.2f}] MW")
        print(f"  - Best strategy range: [{self.x_best.min():.2f}, {self.x_best.max():.2f}] MW")
        print("=" * 60)

        return {
            'gen_power_all': self.gen_power_all,
            'gen_alpha_all': self.gen_alpha_all,
            'x_worst': self.x_worst,
            'x_best': self.x_best,
            'error_worst_t': self.error_worst_t,
            'error_best_t': self.error_best_t,
        }

    def online_dispatch(self, WT_pred: np.ndarray, WT_actual: np.ndarray,
                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Online dispatch phase: compute adaptive dispatch based on real-time observations.

        Parameters:
        -----------
        WT_pred : np.ndarray
            Predicted wind power (T, num_WT)
        WT_actual : np.ndarray
            Actual observed wind power (T, num_WT)
        verbose : bool
            Whether to print detailed information

        Returns:
        --------
        x_real : np.ndarray
            Real-time dispatch (T, num_gen)
        lambda_t : np.ndarray
            Quantile parameters for each time step (T,)
        """
        if self.gen_power_all is None:
            raise ValueError("Offline optimization must be run first. Call solve_offline() method.")

        # Calculate real-time error and its sum
        e_real = WT_actual - WT_pred  # (T, num_WT)
        e_real_sum = e_real.sum(axis=-1)  # (T,)

        # Calculate quantile parameter λ_t
        # λ_t = (e_real_sum - e_best) / (e_worst - e_best)
        denominator = self.error_worst_t - self.error_best_t

        # Handle potential division by zero (if worst == best)
        safe_denominator = np.where(np.abs(denominator) < 1e-6, 1.0, denominator)
        lambda_t = (e_real_sum - self.error_best_t) / safe_denominator

        # Clip λ_t to [0, 1]
        lambda_t = np.clip(lambda_t, 0, 1)

        # Interpolate dispatch: x̃_t = λ_t * ẋ_t + (1-λ_t) * x̂_t
        x_real = np.zeros((self.T, self.num_gen))
        for t in range(self.T):
            x_real[t, :] = (lambda_t[t] * self.x_worst[t, :] +
                           (1 - lambda_t[t]) * self.x_best[t, :])

        if verbose:
            print(f"\n--- Online Dispatch Statistics ---")
            print(f"  Real-time error sum range: [{e_real_sum.min():.2f}, {e_real_sum.max():.2f}] MW")
            print(f"  Lambda (quantile) range: [{lambda_t.min():.4f}, {lambda_t.max():.4f}]")
            print(f"  Mean lambda: {lambda_t.mean():.4f}")
            print(f"  Dispatch range: [{x_real.min():.2f}, {x_real.max():.2f}] MW")
            print("-" * 30)

        return x_real, lambda_t

    def compute_cost(self, x_dispatch: np.ndarray, gen_cost: np.ndarray,
                     gen_cost_quadra: np.ndarray) -> float:
        """
        Compute total generation cost for a given dispatch.

        Parameters:
        -----------
        x_dispatch : np.ndarray
            Generator dispatch (T, num_gen)
        gen_cost : np.ndarray
            Linear cost coefficients (num_gen,)
        gen_cost_quadra : np.ndarray
            Quadratic cost coefficients (num_gen,)

        Returns:
        --------
        cost : float
            Total generation cost
        """
        FC = gen_cost * x_dispatch + gen_cost_quadra * x_dispatch ** 2
        return FC.sum()

    def evaluate_scenario(self, WT_pred: np.ndarray,
                          WT_error_scenario: np.ndarray,
                          gen_cost_quadra: np.ndarray,
                          strategy: str = 'quantile') -> Dict:
        """
        Evaluate a single scenario using different dispatch strategies.

        Parameters:
        -----------
        WT_pred : np.ndarray
            Predicted wind power (T, num_WT)
        WT_error_scenario : np.ndarray
            Wind error scenario (T, num_WT)
        gen_cost_quadra : np.ndarray
            Quadratic cost coefficients (num_gen,)
        strategy : str
            Dispatch strategy ('quantile', 'FICA', 'worst', 'best')

        Returns:
        --------
        results : Dict
            Dictionary containing dispatch and cost information
        """
        WT_actual = WT_pred + WT_error_scenario

        if strategy == 'quantile':
            # Dynamic quantile policy
            x_dispatch, lambda_t = self.online_dispatch(WT_pred, WT_actual, verbose=False)
        elif strategy == 'FICA':
            # Standard FICA with scalar error sum
            e_real_sum = WT_error_scenario.sum(axis=-1)
            x_dispatch = np.zeros((self.T, self.num_gen))
            for t in range(self.T):
                x_dispatch[t, :] = (self.gen_power_all[t, :] -
                                   self.gen_alpha_all[t, :] * e_real_sum[t])
        elif strategy == 'worst':
            # Worst-case strategy
            x_dispatch = self.x_worst.copy()
        elif strategy == 'best':
            # Best-case strategy
            x_dispatch = self.x_best.copy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute cost
        cost = self.compute_cost(x_dispatch, self.gen_cost, gen_cost_quadra)

        # Check feasibility
        feasible = self._check_feasibility(x_dispatch, WT_pred, WT_error_scenario)

        return {
            'x_dispatch': x_dispatch,
            'cost': cost,
            'feasible': feasible,
            'lambda_t': lambda_t if strategy == 'quantile' else None
        }

    def _validate_extreme_strategies(self):
        """Validate that extreme strategies satisfy basic constraints."""
        for t in range(self.T):
            for g in range(self.num_gen):
                # Check capacity limits (allow small violations)
                if self.x_worst[t, g] < self.gen_pmin_individual[g] - 0.01:
                    print(f"  Warning: x_worst[{t},{g}] = {self.x_worst[t,g]:.2f} < Pmin = {self.gen_pmin_individual[g]:.2f}")
                if self.x_worst[t, g] > self.gen_cap_individual[g] + 0.01:
                    print(f"  Warning: x_worst[{t},{g}] = {self.x_worst[t,g]:.2f} > Pmax = {self.gen_cap_individual[g]:.2f}")
                if self.x_best[t, g] < self.gen_pmin_individual[g] - 0.01:
                    print(f"  Warning: x_best[{t},{g}] = {self.x_best[t,g]:.2f} < Pmin = {self.gen_pmin_individual[g]:.2f}")
                if self.x_best[t, g] > self.gen_cap_individual[g] + 0.01:
                    print(f"  Warning: x_best[{t},{g}] = {self.x_best[t,g]:.2f} > Pmax = {self.gen_cap_individual[g]:.2f}")

    def _check_feasibility(self, x_dispatch: np.ndarray, WT_pred: np.ndarray,
                          WT_error_scenario: np.ndarray) -> bool:
        """Check if dispatch satisfies basic capacity constraints."""
        for t in range(self.T):
            for g in range(self.num_gen):
                if x_dispatch[t, g] < self.gen_pmin_individual[g] - 0.1:
                    return False
                if x_dispatch[t, g] > self.gen_cap_individual[g] + 0.1:
                    return False
        return True

    def save(self, filepath: str):
        """Save the policy to disk."""
        data = {
            'T': self.T,
            'num_gen': self.num_gen,
            'num_WT': self.num_WT,
            'epsilon': self.epsilon,
            'theta': self.theta,
            'gen_power_all': self.gen_power_all,
            'gen_alpha_all': self.gen_alpha_all,
            'x_worst': self.x_worst,
            'x_best': self.x_best,
            'error_worst_t': self.error_worst_t,
            'error_best_t': self.error_best_t,
            'gen_cap_individual': self.gen_cap_individual,
            'gen_pmin_individual': self.gen_pmin_individual,
            'gen_cost': self.gen_cost,
        }
        joblib.dump(data, filepath)
        print(f"✓ Policy saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'QuantilePolicy':
        """Load a policy from disk."""
        data = joblib.load(filepath)

        policy = cls(
            T=data['T'],
            num_gen=data['num_gen'],
            num_WT=data['num_WT'],
            epsilon=data['epsilon'],
            theta=data['theta']
        )

        policy.gen_power_all = data['gen_power_all']
        policy.gen_alpha_all = data['gen_alpha_all']
        policy.x_worst = data['x_worst']
        policy.x_best = data['x_best']
        policy.error_worst_t = data['error_worst_t']
        policy.error_best_t = data['error_best_t']
        policy.gen_cap_individual = data['gen_cap_individual']
        policy.gen_pmin_individual = data['gen_pmin_individual']
        policy.gen_cost = data['gen_cost']

        print(f"✓ Policy loaded from {filepath}")
        return policy


def compute_comparison_statistics(costs_quantile: np.ndarray,
                                  costs_FICA: np.ndarray,
                                  costs_worst: np.ndarray,
                                  costs_best: np.ndarray) -> Dict:
    """
    Compute comparison statistics between different strategies.

    Parameters:
    -----------
    costs_quantile : np.ndarray
        Costs from quantile policy (N_scenarios,)
    costs_FICA : np.ndarray
        Costs from standard FICA (N_scenarios,)
    costs_worst : np.ndarray
        Costs from worst-case strategy (N_scenarios,)
    costs_best : np.ndarray
        Costs from best-case strategy (N_scenarios,)

    Returns:
    --------
    stats : Dict
        Dictionary containing comparison statistics
    """
    stats = {
        'quantile': {
            'mean': np.mean(costs_quantile),
            'std': np.std(costs_quantile),
            'min': np.min(costs_quantile),
            'max': np.max(costs_quantile),
            'median': np.median(costs_quantile),
        },
        'FICA': {
            'mean': np.mean(costs_FICA),
            'std': np.std(costs_FICA),
            'min': np.min(costs_FICA),
            'max': np.max(costs_FICA),
            'median': np.median(costs_FICA),
        },
        'worst': {
            'mean': np.mean(costs_worst),
            'std': np.std(costs_worst),
        },
        'best': {
            'mean': np.mean(costs_best),
            'std': np.std(costs_best),
        },
    }

    # Improvement of quantile over FICA
    cost_diff = costs_FICA - costs_quantile
    stats['improvement'] = {
        'mean': np.mean(cost_diff),
        'std': np.std(cost_diff),
        'min': np.min(cost_diff),
        'max': np.max(cost_diff),
        'percentage': np.mean(cost_diff) / np.mean(costs_FICA) * 100,
        'wins': np.sum(cost_diff > 0),
        'total': len(cost_diff),
        'win_rate': np.mean(cost_diff > 0) * 100,
    }

    return stats


def print_comparison_stats(stats: Dict):
    """Print comparison statistics in a formatted way."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    strategies = ['quantile', 'FICA', 'worst', 'best']
    names = {'quantile': 'Quantile Policy', 'FICA': 'Standard FICA',
             'worst': 'Worst-Case', 'best': 'Best-Case'}

    for strat in strategies:
        if strat in stats:
            name = names[strat]
            s = stats[strat]
            print(f"\n{name}:")
            print(f"  Mean cost:  ${s['mean']:,.2f}")
            if 'std' in s:
                print(f"  Std cost:   ${s['std']:,.2f}")
            if 'min' in s:
                print(f"  Min cost:   ${s['min']:,.2f}")
            if 'max' in s:
                print(f"  Max cost:   ${s['max']:,.2f}")
            if 'median' in s:
                print(f"  Median:     ${s['median']:,.2f}")

    if 'improvement' in stats:
        imp = stats['improvement']
        print("\n" + "-" * 70)
        print("Quantile Policy vs Standard FICA:")
        print(f"  Mean improvement:       ${imp['mean']:,.2f}")
        print(f"  Improvement rate:        {imp['percentage']:.2f}%")
        print(f"  Scenarios improved:     {imp['wins']} / {imp['total']}")
        print(f"  Win rate:               {imp['win_rate']:.2f}%")
        print(f"  Improvement range:     [${imp['min']:,.2f}, ${imp['max']:,.2f}]")
        print("=" * 70)
