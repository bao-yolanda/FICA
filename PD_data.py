"""
Data processing and utility functions for PD optimization.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
from WT_error_gen import WT_sce_gen


def check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test,
              P_line_limit, gen_bus_list, WT_bus_list,
              storage_p=None, storage_alpha=None, storage_bus_list=None):
    """Check Joint Chance Constraint (JCC) satisfaction rate.

    Args:
        T: Time horizon
        num_gen: Number of generators
        num_branch: Number of branches
        gen_power_all: Generator power output (T x num_gen)
        gen_alpha_all: Generator AGC factors (T x num_gen)
        load_bus_all: Load at all buses (T x num_bus)
        PTDF: Power Transfer Distribution Factors
        gen_cap_individual: Generator capacity limits (num_gen)
        gen_pmin_individual: Generator minimum generation limits (num_gen)
        WT_pred: Wind power prediction (T x num_WT)
        WT_error_scenarios_test: Wind power error scenarios (N_test x T x num_WT)
        P_line_limit: Line flow limits (num_branch)
        gen_bus_list: Generator bus indices (num_gen)
        WT_bus_list: Wind turbine bus indices (num_WT)
        storage_p: Storage scheduled power (T)
        storage_alpha: Storage AGC factor (T)
        storage_bus_list: Storage bus indices

    Returns:
        satisfied_rate: Ratio of scenarios satisfying all constraints
    """
    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0
    PTDF_gen = PTDF[:, gen_bus_list].T
    PTDF_wind = PTDF[:, WT_bus_list].T
    PTDF_load = PTDF.T  # the load_bus_all is the load at all buses, with shape (T, num_bus)

    # Pmax min constraints
    P_res = []
    for t in range(T):
        for g in range(num_gen):
            gen_power_adjusted = gen_power_all[t, g] - WT_error_scenarios_test.sum(axis=-1)[:, t] * gen_alpha_all[t, g]
            P_res.append(gen_power_adjusted <= gen_cap_individual[g])
            P_res.append(gen_power_adjusted >= gen_pmin_individual[g])

    # Line flow constraints
    L_res = []
    for t in range(T):
        for l in range(num_branch):
            line_flow = ((gen_power_all[t] - gen_alpha_all[t] * WT_error_scenarios_test.sum(axis=-1)[:, t:t+1]) @ PTDF_gen[:, l]
                         + (WT_pred[t] + WT_error_scenarios_test[:, t]) @ PTDF_wind[:, l] - load_bus_all[t] @ PTDF_load[:, l])
            # Add storage contribution to line flow if storage is enabled
            if storage_p is not None and storage_alpha is not None and storage_bus_list is not None:
                storage_power_actual = storage_p[t] + storage_alpha[t] * WT_error_scenarios_test.sum(axis=-1)[:, t]
                for s_bus in storage_bus_list:
                    line_flow += storage_power_actual * PTDF[s_bus, l]
            L_res.append(line_flow <= P_line_limit[l])
            L_res.append(line_flow >= -P_line_limit[l])

    res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))
    return satisfied_rate


class NetworkGenerator:
    """Generate and configure power network data."""

    def __init__(self, network_name='case5'):
        self.network_name = network_name
        self.network = self._load_network(network_name)

    def _load_network(self, network_name):
        """Load standard pandapower network."""
        network_dict = {
            'case118': ppnw.case118(),
            'case300': ppnw.case300(),
            'case24_ieee_rts': ppnw.case24_ieee_rts(),
            'case5': ppnw.case5(),
            'case4gs': ppnw.case4gs(),
            'case_ieee30': ppnw.case_ieee30()
        }
        return network_dict[network_name]

    def get_network_info(self, num_gen, num_WT, load_scaling_factor=1, Tstart=0, T=24, gen_cap_total_prop=1):
        """Generate complete network information for optimization.

        Args:
            num_gen: Number of generators
            num_WT: Number of wind turbines
            load_scaling_factor: Load scaling factor
            Tstart: Start time index
            T: Time horizon
            gen_cap_total_prop: Total generation capacity proportion to load

        Returns:
            NetworkInfo: Network configuration data
        """
        from PD_microgrid import NetworkInfo

        # Run power flow to get PTDF
        pp.rundcpp(self.network)
        _, ppci = _pd2ppc(self.network)
        bus_info = ppci['bus']
        branch_info = ppci['branch']
        PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info, using_sparse_solver=False)

        num_branch = len(branch_info)

        # Load data
        load_bus_size = bus_info[:, 2] * load_scaling_factor
        load_total = np.sum(load_bus_size)

        # Generate load curve
        load_bus_all = self._generate_load_curve(load_bus_size, Tstart, T)

        # Generator configuration
        gen_cap_individual, gen_pmin_individual = self._generate_generator_config(
            num_gen, load_total, gen_cap_total_prop
        )

        # Generator cost
        gen_cost = np.random.uniform(23.13, 57.03, num_gen)
        gen_cost_quadra = np.random.uniform(0.002, 0.008, num_gen)

        # Generator and wind locations
        rng = np.random.RandomState(0)
        bus_list = np.arange(bus_info.shape[0])
        gen_bus_list = rng.choice(bus_list, num_gen, replace=True)
        WT_bus_list = rng.choice(bus_list, num_WT, replace=True)

        # Line limits
        P_line_limit = np.abs(ppci['branch'][:, 5])
        P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

        return NetworkInfo(
            name=self.network_name,
            num_gen=num_gen,
            num_WT=num_WT,
            num_branch=num_branch,
            gen_bus_list=gen_bus_list,
            WT_bus_list=WT_bus_list,
            PTDF=PTDF,
            load_bus_all=load_bus_all,
            gen_cap_individual=gen_cap_individual,
            gen_pmin_individual=gen_pmin_individual,
            gen_cost=gen_cost,
            gen_cost_quadra=gen_cost_quadra,
            P_line_limit=P_line_limit
        )

    def _generate_load_curve(self, load_bus_size, Tstart, T):
        """Generate load curve from normalized data."""
        import os

        load_location = os.path.join(os.getcwd(), 'data', 'UK_norm_load_curve_highest.npy')
        network_load = np.load(load_location)

        # Convert half-hourly to hourly
        network_load = np.mean(np.vstack([network_load[::2], network_load[1::2]]), axis=0)

        # Duplicate for 2 days
        network_load = np.tile(network_load, 2)
        network_load = network_load[Tstart:Tstart+T]

        # Apply to all buses
        load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

        return load_bus_all

    def _generate_generator_config(self, num_gen, load_total, gen_cap_total_prop):
        """Generate generator capacity and minimum limits."""
        rng = np.random.RandomState(0)

        gen_cap_total = load_total * gen_cap_total_prop
        gen_cap_individual = gen_cap_total / num_gen
        gen_cap_individual = rng.uniform(0.6, 1.4, num_gen) * gen_cap_individual
        gen_pmin_individual = 0.1 * gen_cap_individual

        return gen_cap_individual, gen_pmin_individual


class WindDataGenerator:
    """Generate wind power scenarios."""

    def __init__(self, num_WT, load_total, wind_ratio=0.6):
        self.num_WT = num_WT
        self.load_total = load_total
        self.wind_ratio = wind_ratio

    def generate(self, Tstart=0, T=24, N_train=1000, N_test=5000):
        """Generate wind power prediction and error scenarios.

        Args:
            Tstart: Start time index
            T: Time horizon
            N_train: Number of training scenarios
            N_test: Number of testing scenarios

        Returns:
            WindData: Wind power data structure
        """
        from PD_microgrid import WindData

        WT_total = self.wind_ratio * self.load_total
        WT_individual = WT_total / self.num_WT

        # Generate scenarios
        WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(
            self.num_WT, N_train + N_test
        )

        # Scale and slice
        WT_pred = WT_pred[Tstart:Tstart+T] * WT_individual
        WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual
        WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual

        # Split into train and test
        WT_error_scenarios_train = WT_error_scenarios[:N_train]
        WT_error_scenarios_test = WT_error_scenarios[N_train:]

        return WindData(
            pred=WT_pred,
            error_scenarios_train=WT_error_scenarios_train,
            error_scenarios_test=WT_error_scenarios_test
        )


def create_optimization_data(network_name, num_gen, num_WT, Tstart=0, T=24, load_scaling_factor=1,
                            gen_cap_total_prop=1, N_train=1000, N_test=5000):
    """Create complete optimization data in one function.

    Args:
        network_name: Network name (e.g., 'case5', 'case118')
        num_gen: Number of generators
        num_WT: Number of wind turbines
        Tstart: Start time index
        T: Time horizon
        load_scaling_factor: Load scaling factor
        gen_cap_total_prop: Generation capacity proportion
        N_train: Number of training scenarios
        N_test: Number of testing scenarios

    Returns:
        tuple: (NetworkInfo, WindData) containing all necessary data
    """
    # Generate network info
    network_gen = NetworkGenerator(network_name)
    network_info = network_gen.get_network_info(
        num_gen, num_WT, load_scaling_factor, Tstart, T, gen_cap_total_prop
    )

    # Generate wind data
    load_total = np.sum(network_info.load_bus_all[0])
    wind_gen = WindDataGenerator(num_WT, load_total)
    wind_data = wind_gen.generate(Tstart, T, N_train, N_test)

    return network_info, wind_data


if __name__ == '__main__':
    # Example usage
    print("Generating test data...")

    # Create network info
    network_info, wind_data = create_optimization_data(
        network_name='case5',
        num_gen=2,
        num_WT=2,
        T=24
    )

    print(f"Network: {network_info.name}")
    print(f"Generators: {network_info.num_gen}")
    print(f"Wind turbines: {network_info.num_WT}")
    print(f"Branches: {network_info.num_branch}")
    print(f"Load total: {np.sum(network_info.load_bus_all[0]):.2f} MW")
    print(f"Gen capacity total: {np.sum(network_info.gen_cap_individual):.2f} MW")
    print(f"Wind prediction shape: {wind_data.pred.shape}")
    print(f"Wind error train shape: {wind_data.error_scenarios_train.shape}")
    print(f"Wind error test shape: {wind_data.error_scenarios_test.shape}")
