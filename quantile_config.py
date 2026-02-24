"""
Configuration file for Quantile Policy Evaluation

This file contains default parameters and configurations for running
the quantile policy evaluation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class QuantileConfig:
    """Configuration for quantile policy experiments."""

    # System parameters
    num_gen: int = 2
    num_WT: int = 2
    T: int = 24

    # Optimization parameters
    N_WDR: int = 100
    epsilon: float = 0.05
    theta: float = 0.15
    norm_ord: int = 1
    load_scaling_factor: float = 1.0

    # Scenario parameters
    N_samples_train: int = 1000
    N_samples_test: int = 1000

    # Evaluation parameters
    N_test: int = 100
    gurobi_seed: int = 0
    Tstart: int = 0

    # Solver parameters
    MIPGap: float = 0.001
    bigM: float = 1e5
    thread: int = 4

    # Output parameters
    save_dir: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.save_dir is None:
            self.save_dir = f"figure/quantile_evaluation/gen{self.num_gen}_WT{self.num_WT}"

    @classmethod
    def small_test(cls) -> 'QuantileConfig':
        """Configuration for quick testing (1-2 minutes)."""
        return cls(
            num_gen=2,
            num_WT=2,
            T=12,
            N_WDR=50,
            epsilon=0.05,
            theta=0.15,
            norm_ord=1,
            N_samples_train=500,
            N_samples_test=500,
            N_test=20,
        )

    @classmethod
    def medium_test(cls) -> 'QuantileConfig':
        """Configuration for medium-scale testing (5-10 minutes)."""
        return cls(
            num_gen=2,
            num_WT=2,
            T=24,
            N_WDR=100,
            epsilon=0.05,
            theta=0.15,
            norm_ord=1,
            N_samples_train=1000,
            N_samples_test=1000,
            N_test=100,
        )

    @classmethod
    def large_test(cls) -> 'QuantileConfig':
        """Configuration for large-scale testing (30-60 minutes)."""
        return cls(
            num_gen=5,
            num_WT=3,
            T=24,
            N_WDR=200,
            epsilon=0.05,
            theta=0.15,
            norm_ord=1,
            N_samples_train=2000,
            N_samples_test=2000,
            N_test=500,
        )

    @classmethod
    def sensitivity_epsilon(cls) -> 'QuantileConfig':
        """Configuration for epsilon sensitivity analysis."""
        return cls(
            num_gen=2,
            num_WT=2,
            T=24,
            N_WDR=100,
            epsilon=0.03,  # Vary this: 0.01, 0.03, 0.05, 0.07, 0.10
            theta=0.15,
            norm_ord=1,
            N_test=100,
        )

    @classmethod
    def sensitivity_theta(cls) -> 'QuantileConfig':
        """Configuration for theta sensitivity analysis."""
        return cls(
            num_gen=2,
            num_WT=2,
            T=24,
            N_WDR=100,
            epsilon=0.05,
            theta=0.13,  # Vary this: 0.10, 0.13, 0.15, 0.17, 0.20
            norm_ord=1,
            N_test=100,
        )

    @classmethod
    def sensitivity_system(cls) -> 'QuantileConfig':
        """Configuration for system size sensitivity analysis."""
        return cls(
            num_gen=3,  # Vary this: 2, 3, 5, 10
            num_WT=2,
            T=24,
            N_WDR=100,
            epsilon=0.05,
            theta=0.15,
            norm_ord=1,
            N_test=100,
        )

    def print_config(self):
        """Print current configuration."""
        print("=" * 70)
        print("QUANTILE POLICY CONFIGURATION")
        print("=" * 70)
        print(f"System Parameters:")
        print(f"  - Generators: {self.num_gen}")
        print(f"  - Wind farms: {self.num_WT}")
        print(f"  - Time horizon: {self.T} hours")
        print(f"\nOptimization Parameters:")
        print(f"  - N_WDR: {self.N_WDR}")
        print(f"  - Risk level (epsilon): {self.epsilon}")
        print(f"  - Wasserstein radius (theta): {self.theta}")
        print(f"  - Norm order: {self.norm_ord}")
        print(f"  - Load scaling: {self.load_scaling_factor}")
        print(f"\nScenario Parameters:")
        print(f"  - Training samples: {self.N_samples_train}")
        print(f"  - Test samples: {self.N_samples_test}")
        print(f"\nEvaluation Parameters:")
        print(f"  - Test scenarios: {self.N_test}")
        print(f"  - Gurobi seed: {self.gurobi_seed}")
        print(f"  - Start time: {self.Tstart}")
        print(f"\nSolver Parameters:")
        print(f"  - MIP Gap: {self.MIPGap}")
        print(f"  - Big M: {self.bigM}")
        print(f"  - Threads: {self.thread}")
        print(f"\nOutput:")
        print(f"  - Save directory: {self.save_dir}")
        print("=" * 70)


# Example usage
if __name__ == '__main__':
    # Create default configuration
    config = QuantileConfig()
    config.print_config()

    # Create small test configuration
    small_config = QuantileConfig.small_test()
    print("\nSmall Test Configuration:")
    small_config.print_config()

    # Create sensitivity analysis configurations
    eps_configs = [QuantileConfig.sensitivity_epsilon() for _ in range(5)]
    eps_values = [0.01, 0.03, 0.05, 0.07, 0.10]
    for cfg, eps in zip(eps_configs, eps_values):
        cfg.epsilon = eps
        print(f"\nEpsilon = {eps}:")
        print(f"  Risk level: {cfg.epsilon}")
