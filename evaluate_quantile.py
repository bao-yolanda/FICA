"""
Evaluation Module for Dynamic Alpha Quantile Policy
====================================================

This module provides comprehensive evaluation and comparison between the
dynamic quantile policy and standard FICA approach.

Features:
    - Runs multiple test scenarios
    - Compares costs across different strategies
    - Visualizes results
    - Saves evaluation data for further analysis
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Tuple, List
import pandas as pd

from quantile_policy import QuantilePolicy, compute_comparison_statistics, print_comparison_stats
from PD import solve_PD_instance, check_JCC
from WT_error_gen import WT_sce_gen


class QuantileEvaluator:
    """
    Comprehensive evaluator for quantile policy vs standard FICA.
    """

    def __init__(self, num_gen: int = 2, num_WT: int = 2, T: int = 24,
                 N_WDR: int = 100, epsilon: float = 0.05, theta: float = 0.15,
                 norm_ord: int = 1, load_scaling_factor: float = 1.0,
                 N_samples_train: int = 1000, N_samples_test: int = 1000):
        """
        Initialize evaluator.

        Parameters:
        -----------
        num_gen : int
            Number of generators
        num_WT : int
            Number of wind farms
        T : int
            Time horizon
        N_WDR : int
            Number of Wasserstein scenarios
        epsilon : float
            Risk level
        theta : float
            Wasserstein radius
        norm_ord : int
            Norm order (1 or 2)
        load_scaling_factor : float
            Load scaling factor
        N_samples_train : int
            Number of training scenarios
        N_samples_test : int
            Number of test scenarios
        """
        self.num_gen = num_gen
        self.num_WT = num_WT
        self.T = T
        self.N_WDR = N_WDR
        self.epsilon = epsilon
        self.theta = theta
        self.norm_ord = norm_ord
        self.load_scaling_factor = load_scaling_factor
        self.N_samples_train = N_samples_train
        self.N_samples_test = N_samples_test

        # Storage for data and results
        self.WT_pred = None
        self.WT_error_scenarios_train = None
        self.WT_error_scenarios_test = None
        self.load_bus_all = None
        self.PTDF = None
        self.gen_cap_individual = None
        self.gen_pmin_individual = None
        self.gen_cost = None
        self.gen_cost_quadra = None
        self.P_line_limit = None
        self.gen_bus_list = None
        self.WT_bus_list = None
        self.num_branch = None

        # Results
        self.quantile_policy = None
        self.evaluation_results = None

    def setup_data(self, Tstart: int = 0, method: str = 'FICA',
                   gurobi_seed: int = 0):
        """
        Setup data structures for evaluation.

        This replicates the data setup from PD.py to ensure consistency.
        """
        print("=" * 70)
        print("Setting up data structures...")
        print("=" * 70)

        import pandapower as pp
        import pandapower.networks as ppnw
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pd2ppc import _pd2ppc

        network_name = 'case5'
        gen_cap_total_prop = 1
        rng = np.random.RandomState(gurobi_seed)
        rng_fixed = np.random.RandomState(0)

        # Load network
        network_dict = {
            'case118': ppnw.case118(),
            'case300': ppnw.case300(),
            'case24_ieee_rts': ppnw.case24_ieee_rts(),
            'case5': ppnw.case5(),
            'case4gs': ppnw.case4gs(),
            'case_ieee30': ppnw.case_ieee30()
        }
        network = network_dict[network_name]

        # Load data
        load_location = os.path.join(os.getcwd(), 'data', 'UK_norm_load_curve_highest.npy')
        network_load = np.load(load_location)
        network_load = np.mean(np.vstack([network_load[::2], network_load[1::2]]), axis=0)
        network_load = np.tile(network_load, 2)
        network_load = network_load[Tstart:Tstart+self.T]

        # Power flow data
        pp.rundcpp(network)
        _, ppci = _pd2ppc(network)
        bus_info = ppci['bus']
        branch_info = ppci['branch']
        self.PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                            using_sparse_solver=False)

        self.num_branch = len(branch_info)

        # Load data
        load_bus_size = bus_info[:, 2] * self.load_scaling_factor
        load_total = np.sum(load_bus_size)
        self.load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

        # Generator capacity
        gen_cap_total = load_total * gen_cap_total_prop
        self.gen_cap_individual = rng_fixed.uniform(0.6, 1.4, self.num_gen) * (gen_cap_total / self.num_gen)
        self.gen_pmin_individual = 0.1 * self.gen_cap_individual

        # Generator cost
        self.gen_cost = rng.uniform(23.13, 57.03, self.num_gen)
        self.gen_cost_quadra = rng.uniform(0.002, 0.008, self.num_gen)

        # Generator and wind locations
        bus_list = np.arange(bus_info.shape[0])
        self.gen_bus_list = rng_fixed.choice(bus_list, self.num_gen, replace=True)
        self.WT_bus_list = rng_fixed.choice(bus_list, self.num_WT, replace=True)

        # Line limits
        self.P_line_limit = np.abs(ppci['branch'][:, 5])
        self.P_line_limit = np.clip(self.P_line_limit, 0, 2 * load_total)

        # Wind data
        WT_total = 0.6 * load_total
        WT_individual = WT_total / self.num_WT
        WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(
            self.num_WT, self.N_samples_train + self.N_samples_test
        )

        self.WT_pred = WT_pred[Tstart:Tstart+self.T] * WT_individual
        WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+self.T] * WT_individual
        WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+self.T] * WT_individual

        self.WT_error_scenarios_train = WT_error_scenarios[:self.N_samples_train]
        self.WT_error_scenarios_test = WT_error_scenarios[self.N_samples_train:]

        print(f"✓ Data setup completed")
        print(f"  - Time horizon: {self.T} hours")
        print(f"  - Generators: {self.num_gen}")
        print(f"  - Wind farms: {self.num_WT}")
        print(f"  - Training scenarios: {self.N_samples_train}")
        print(f"  - Test scenarios: {self.N_samples_test}")
        print(f"  - Load total: {load_total:.2f} MW")
        print("=" * 70)

    def run_offline_optimization(self) -> Dict:
        """
        Run offline optimization using FICA.

        Returns:
        --------
        results : Dict
            Optimization results including gen_power_all and gen_alpha_all
        """
        print("\n" + "=" * 70)
        print("Running Offline FICA Optimization...")
        print("=" * 70)

        # Import here to avoid issues if PD.py has been modified
        from PD import solve_PD

        MIPGap = 0.001
        bigM = 1e5
        log_file_name = None
        thread = 4
        gurobi_seed = 0
        rng = np.random.RandomState(gurobi_seed)

        input_param_dict = {
            'T': self.T, 'num_gen': self.num_gen, 'num_WT': self.num_WT,
            'num_branch': self.num_branch,
            'load_bus_all': self.load_bus_all, 'PTDF': self.PTDF,
            'gen_cap_individual': self.gen_cap_individual,
            'gen_pmin_individual': self.gen_pmin_individual,
            'WT_pred': self.WT_pred,
            'WT_error_scenarios_train': self.WT_error_scenarios_train,
            'P_line_limit': self.P_line_limit,
            'gen_bus_list': self.gen_bus_list, 'WT_bus_list': self.WT_bus_list,
            'N_WDR': self.N_WDR, 'epsilon': self.epsilon, 'theta': self.theta,
            'thread': thread,
            'method': 'FICA', 'MIPGap': MIPGap, 'gen_cost': self.gen_cost,
            'gen_cost_quadra': self.gen_cost_quadra, 'bigM': bigM,
            'gurobi_seed': gurobi_seed, 'log_file_name': log_file_name,
            'rng': rng, "norm_ord": self.norm_ord
        }

        t_start = time.time()
        prob, gen_power_all, gen_alpha_all = solve_PD(**input_param_dict)
        t_solve = time.time() - t_start

        gen_power_all = gen_power_all.X
        gen_alpha_all = gen_alpha_all.X

        # Check JCC satisfaction rate
        satisfied_rate = check_JCC(
            self.T, self.num_gen, self.num_branch,
            gen_power_all, gen_alpha_all, self.load_bus_all, self.PTDF,
            self.gen_cap_individual, self.gen_pmin_individual,
            self.WT_pred, self.WT_error_scenarios_test,
            self.P_line_limit, self.gen_bus_list, self.WT_bus_list
        )

        print(f"\n✓ FICA optimization completed")
        print(f"  - Objective value: ${prob.objVal:,.2f}")
        print(f"  - Solve time: {t_solve:.2f} seconds")
        print(f"  - JCC satisfaction rate: {satisfied_rate*100:.2f}%")
        print("=" * 70)

        return {
            'gen_power_all': gen_power_all,
            'gen_alpha_all': gen_alpha_all,
            'objective_value': prob.objVal,
            'solve_time': t_solve,
            'jcc_satisfaction_rate': satisfied_rate,
        }

    def initialize_quantile_policy(self, offline_results: Dict):
        """
        Initialize the quantile policy with offline results.

        Parameters:
        -----------
        offline_results : Dict
            Results from run_offline_optimization
        """
        print("\n" + "=" * 70)
        print("Initializing Quantile Policy...")
        print("=" * 70)

        self.quantile_policy = QuantilePolicy(
            T=self.T,
            num_gen=self.num_gen,
            num_WT=self.num_WT,
            epsilon=self.epsilon,
            theta=self.theta
        )

        # Solve offline optimization for quantile policy
        self.quantile_policy.solve_offline(
            gen_power_all=offline_results['gen_power_all'],
            gen_alpha_all=offline_results['gen_alpha_all'],
            WT_error_scenarios_train=self.WT_error_scenarios_train,
            gen_cap_individual=self.gen_cap_individual,
            gen_pmin_individual=self.gen_pmin_individual,
            gen_cost=self.gen_cost
        )

        print("=" * 70)

    def evaluate_all_scenarios(self, N_test: int = None) -> Dict:
        """
        Evaluate all test scenarios.

        Parameters:
        -----------
        N_test : int, optional
            Number of test scenarios to evaluate (default: all)

        Returns:
        --------
        results : Dict
            Comprehensive evaluation results
        """
        if N_test is None:
            N_test = min(self.N_samples_test, 500)  # Limit to 500 for efficiency

        print("\n" + "=" * 70)
        print(f"Evaluating {N_test} test scenarios...")
        print("=" * 70)

        costs_quantile = np.zeros(N_test)
        costs_FICA = np.zeros(N_test)
        costs_worst = np.zeros(N_test)
        costs_best = np.zeros(N_test)
        lambda_all = np.zeros((N_test, self.T))
        feasibility_quantile = np.zeros(N_test, dtype=bool)
        feasibility_FICA = np.zeros(N_test, dtype=bool)

        # Sample scenarios
        test_indices = np.random.choice(
            self.WT_error_scenarios_test.shape[0],
            N_test, replace=False
        )

        for i, idx in enumerate(test_indices):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{N_test} ({100*(i+1)/N_test:.1f}%)")

            WT_error_scenario = self.WT_error_scenarios_test[idx]

            # Evaluate quantile policy
            result_quantile = self.quantile_policy.evaluate_scenario(
                self.WT_pred, WT_error_scenario,
                self.gen_cost_quadra, strategy='quantile'
            )
            costs_quantile[i] = result_quantile['cost']
            feasibility_quantile[i] = result_quantile['feasible']
            lambda_all[i, :] = result_quantile['lambda_t']

            # Evaluate standard FICA
            result_FICA = self.quantile_policy.evaluate_scenario(
                self.WT_pred, WT_error_scenario,
                self.gen_cost_quadra, strategy='FICA'
            )
            costs_FICA[i] = result_FICA['cost']
            feasibility_FICA[i] = result_FICA['feasible']

            # Evaluate extreme strategies
            result_worst = self.quantile_policy.evaluate_scenario(
                self.WT_pred, WT_error_scenario,
                self.gen_cost_quadra, strategy='worst'
            )
            costs_worst[i] = result_worst['cost']

            result_best = self.quantile_policy.evaluate_scenario(
                self.WT_pred, WT_error_scenario,
                self.gen_cost_quadra, strategy='best'
            )
            costs_best[i] = result_best['cost']

        # Compute statistics
        stats = compute_comparison_statistics(
            costs_quantile, costs_FICA, costs_worst, costs_best
        )

        # Add feasibility rates
        stats['quantile']['feasibility_rate'] = np.mean(feasibility_quantile) * 100
        stats['FICA']['feasibility_rate'] = np.mean(feasibility_FICA) * 100

        self.evaluation_results = {
            'costs_quantile': costs_quantile,
            'costs_FICA': costs_FICA,
            'costs_worst': costs_worst,
            'costs_best': costs_best,
            'lambda_all': lambda_all,
            'feasibility_quantile': feasibility_quantile,
            'feasibility_FICA': feasibility_FICA,
            'stats': stats,
            'test_indices': test_indices,
        }

        print("\n" + "=" * 70)
        print("✓ Evaluation completed")
        print("=" * 70)
        print_comparison_stats(stats)

        return self.evaluation_results

    def save_results(self, filepath: str):
        """Save evaluation results to disk."""
        if self.evaluation_results is None:
            raise ValueError("No evaluation results to save. Run evaluate_all_scenarios() first.")

        results_to_save = {
            'parameters': {
                'num_gen': self.num_gen,
                'num_WT': self.num_WT,
                'T': self.T,
                'N_WDR': self.N_WDR,
                'epsilon': self.epsilon,
                'theta': self.theta,
                'norm_ord': self.norm_ord,
                'load_scaling_factor': self.load_scaling_factor,
            },
            'results': self.evaluation_results
        }

        joblib.dump(results_to_save, filepath)
        print(f"\n✓ Results saved to {filepath}")

    def visualize_results(self, save_dir: str = None):
        """
        Visualize evaluation results.

        Parameters:
        -----------
        save_dir : str, optional
            Directory to save figures
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results to visualize. Run evaluate_all_scenarios() first.")

        results = self.evaluation_results
        stats = results['stats']

        # Create save directory if specified
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'figure', 'quantile_evaluation')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Figure 1: Cost distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(results['costs_quantile'], bins=30, alpha=0.6, label='Quantile Policy', color='blue')
        ax.hist(results['costs_FICA'], bins=30, alpha=0.6, label='Standard FICA', color='red')
        ax.axvline(stats['quantile']['mean'], color='blue', linestyle='--', linewidth=2)
        ax.axvline(stats['FICA']['mean'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Generation Cost ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Cost Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Box plot
        ax = axes[1]
        data_to_plot = [results['costs_quantile'], results['costs_FICA']]
        bp = ax.boxplot(data_to_plot, labels=['Quantile Policy', 'Standard FICA'],
                       patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Generation Cost ($)', fontsize=12)
        ax.set_title('Cost Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_name = os.path.join(save_dir, 'cost_distribution.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {fig_name}")

        # Figure 2: Lambda distribution over time
        fig, ax = plt.subplots(figsize=(12, 5))

        lambda_mean = results['lambda_all'].mean(axis=0)
        lambda_std = results['lambda_all'].std(axis=0)

        x = np.arange(self.T)
        ax.plot(x, lambda_mean, 'b-', linewidth=2, label='Mean λ')
        ax.fill_between(x, lambda_mean - lambda_std, lambda_mean + lambda_std,
                       alpha=0.3, label='±1 Std Dev')

        ax.set_xlabel('Time (hour)', fontsize=12)
        ax.set_ylabel('Quantile Parameter λ', fontsize=12)
        ax.set_title('Quantile Parameter Distribution Over Time', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, self.T - 0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_name = os.path.join(save_dir, 'lambda_distribution.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {fig_name}")

        # Figure 3: Cost improvement distribution
        fig, ax = plt.subplots(figsize=(10, 5))

        cost_diff = results['costs_FICA'] - results['costs_quantile']

        ax.hist(cost_diff, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.axvline(stats['improvement']['mean'], color='blue', linestyle='-',
                  linewidth=2, label=f"Mean: ${stats['improvement']['mean']:.2f}")

        ax.set_xlabel('Cost Improvement (FICA - Quantile) ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Cost Improvement', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_name = os.path.join(save_dir, 'cost_improvement.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {fig_name}")

        # Figure 4: Summary table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')

        table_data = [
            ['Metric', 'Quantile Policy', 'Standard FICA', 'Improvement'],
            ['Mean Cost', f"${stats['quantile']['mean']:,.2f}", f"${stats['FICA']['mean']:,.2f}", f"${stats['improvement']['mean']:,.2f}"],
            ['Std Cost', f"${stats['quantile']['std']:,.2f}", f"${stats['FICA']['std']:,.2f}", f"{stats['improvement']['percentage']:.2f}%"],
            ['Median Cost', f"${stats['quantile']['median']:,.2f}", f"${stats['FICA']['median']:,.2f}", f"-"],
            ['Min Cost', f"${stats['quantile']['min']:,.2f}", f"${stats['FICA']['min']:,.2f}", f"-"],
            ['Max Cost', f"${stats['quantile']['max']:,.2f}", f"${stats['FICA']['max']:,.2f}", f"-"],
            ['Feasibility Rate', f"{stats['quantile']['feasibility_rate']:.2f}%", f"{stats['FICA']['feasibility_rate']:.2f}%", f"-"],
            ['Win Rate', f"-", f"-", f"{stats['improvement']['win_rate']:.2f}%"],
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Color the header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Performance Summary', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        fig_name = os.path.join(save_dir, 'summary_table.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {fig_name}")

        print(f"\n✓ All visualizations saved to {save_dir}")


def run_full_evaluation(num_gen: int = 2, num_WT: int = 2, T: int = 24,
                       N_WDR: int = 100, epsilon: float = 0.05,
                       theta: float = 0.15, norm_ord: int = 1,
                       N_test: int = 100, save_dir: str = None):
    """
    Run the complete evaluation pipeline.

    Parameters:
    -----------
    num_gen : int
        Number of generators
    num_WT : int
        Number of wind farms
    T : int
        Time horizon
    N_WDR : int
        Number of Wasserstein scenarios
    epsilon : float
        Risk level
    theta : float
        Wasserstein radius
    norm_ord : int
        Norm order
    N_test : int
        Number of test scenarios
    save_dir : str
        Directory to save results
    """
    print("\n" + "=" * 70)
    print("DYNAMIC ALPHA QUANTILE POLICY - FULL EVALUATION")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  - Generators: {num_gen}")
    print(f"  - Wind farms: {num_WT}")
    print(f"  - Time horizon: {T} hours")
    print(f"  - Risk level: {epsilon}")
    print(f"  - Wasserstein radius: {theta}")
    print(f"  - Test scenarios: {N_test}")
    print("=" * 70)

    # Create evaluator
    evaluator = QuantileEvaluator(
        num_gen=num_gen, num_WT=num_WT, T=T,
        N_WDR=N_WDR, epsilon=epsilon, theta=theta,
        norm_ord=norm_ord
    )

    # Setup data
    evaluator.setup_data()

    # Run offline optimization
    offline_results = evaluator.run_offline_optimization()

    # Initialize quantile policy
    evaluator.initialize_quantile_policy(offline_results)

    # Evaluate all scenarios
    evaluation_results = evaluator.evaluate_all_scenarios(N_test=N_test)

    # Save results
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'figure', 'quantile_evaluation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_file = os.path.join(save_dir, f'eval_results_gen{num_gen}_WT{num_WT}_T{T}.pkl')
    evaluator.save_results(results_file)

    # Visualize results
    evaluator.visualize_results(save_dir=save_dir)

    # Save policy
    policy_file = os.path.join(save_dir, f'policy_gen{num_gen}_WT{num_WT}_T{T}.pkl')
    evaluator.quantile_policy.save(policy_file)

    print("\n" + "=" * 70)
    print("✓ Full evaluation completed successfully!")
    print("=" * 70)

    return evaluator, evaluation_results


if __name__ == '__main__':
    # Run full evaluation with default parameters
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
