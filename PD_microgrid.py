import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
import os
from gurobipy import GRB
from WT_error_gen import WT_sce_gen
from scipy.linalg import norm
from PD_model import solve_PD, solve_PD_actual
from PD_param import PDParams
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from PD_plot import plot_paper, plot_power_balance
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

def check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test,
              P_line_limit, gen_bus_list, WT_bus_list, storage_p, storage_alpha, storage_bus_list):
    # WT_error_scenarios_test has the shape of (N_samples_test, T, num_WT)

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0
    PTDF_gen = PTDF[:, gen_bus_list].T
    PTDF_wind = PTDF[:, WT_bus_list].T
    PTDF_load = PTDF.T

    # Pmax min constraints (multiple generators)
    P_res = []
    for t in range(T):
        for g in range(num_gen):
            gen_power_adjusted = gen_power_all[t, g] - WT_error_scenarios_test.sum(axis=-1)[:, t] * gen_alpha_all[t, g]
            P_res.append(gen_power_adjusted <= gen_cap_individual[g])
            P_res.append(gen_power_adjusted >= gen_pmin_individual[g])

    # Line flow constraints
    L_res = []
    storage_bus = storage_bus_list[0]
    PTDF_storage = PTDF[:, storage_bus]  # 1D array
    for t in range(T):
        for l in range(num_branch):
            line_flow = ((gen_power_all[t] - gen_alpha_all[t] * WT_error_scenarios_test.sum(axis=-1)[:, t:t+1]) @ PTDF_gen[:, l]
                         + (WT_pred[t] + WT_error_scenarios_test[:, t]) @ PTDF_wind[:, l]
                         - load_bus_all[t] @ PTDF_load[:, l]
                         + PTDF_storage[l] * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_test.sum(axis=-1)[:, t]))
            L_res.append(line_flow <= P_line_limit[l])
            L_res.append(line_flow >= -P_line_limit[l])

    res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))
    return satisfied_rate



def solve_PD_instance(params: PDParams):
    '''Solve power dispatch with given parameters'''
    T = params.T
    Tstart = params.Tstart
    num_gen = params.num_gen
    num_WT = params.num_WT
    method = params.method
    N_WDR = params.N_WDR
    epsilon = params.epsilon
    theta = params.theta
    load_scaling_factor = params.load_scaling_factor
    storage_capacity = params.storage_capacity
    storage_power = params.storage_power

    N_samples_train = params.N_samples_train
    N_samples_test = params.N_samples_test
    gurobi_seed = params.gurobi_seed

    network_name = params.network_name
    gen_cap_total_prop = params.gen_cap_total_prop
    #------------------

    network_dict = {'case118': ppnw.case118(),
                    'case300': ppnw.case300(),
                    'case24_ieee_rts': ppnw.case24_ieee_rts(),
                    'case5': ppnw.case5(),
                    'case4gs': ppnw.case4gs(),
                    'case_ieee30': ppnw.case_ieee30()}

    seed = gurobi_seed
    rng = np.random.RandomState(seed)
    rng_fixed = np.random.RandomState(0)  # this is to avoid too much randomness that requires too many runs to have stable results

    # load network model
    network = network_dict[network_name]

    # load network load data
    load_location = os.path.join(os.getcwd(), 'data', 'UK_norm_load_curve_highest.npy')
    network_load = np.load(load_location)
    # the network load is at half-hourly resolution, we need to average the consceutive time steps to get hourly resolution
    network_load = np.mean(np.vstack([network_load[::2],
                                      network_load[1::2]]), axis=0)
    # duplicate the network load to make it two days
    network_load = np.tile(network_load, 2)
    network_load = network_load[Tstart:Tstart+T]

    # -------------------------------------
    pp.rundcpp(network)
    _, ppci = _pd2ppc(network)
    bus_info = ppci['bus']
    branch_info = ppci['branch']
    PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                    using_sparse_solver=False)

    num_branch = len(branch_info)

    # get load info
    load_bus_size = bus_info[:, 2] * load_scaling_factor

    load_total = np.sum(load_bus_size)
    print('load_total:', load_total)
    # we then get the load curves at all buses, using the network_load curve
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    ###### set generator capacity
    gen_cap_total = load_total * gen_cap_total_prop  # the total generation capacity
    print('gen_cap_total:', gen_cap_total)
    gen_cap_individual = gen_cap_total / num_gen  # the individual generation capacity
    # add some randomness when assigning the generation capacity to each generator
    gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
    gen_pmin_individual = 0.1 * gen_cap_individual  # the individual minimum generation capacity. 

    # generator cost parameters
    gen_cost = rng.uniform(23.13, 57.03, num_gen)  # the cost of gas generators (USD/MWh)
    gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)  # the quadratic cost of gas generators (USD/MWh^2)

    # get generator locations
    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
    WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)

    # get line info
    P_line_limit = np.abs(ppci['branch'][:, 5])  # the line flow limit
    # clip on 2 times of the total load to avoid numerical issues
    P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

    WT_total = params.WT_total_ratio * load_total
    print("Total wind: ", WT_total)
    WT_individual = WT_total / num_WT
    # load the wind power scenarios, which is decomposed into prediction and error scenarios
    WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(num_WT, N_samples_train + N_samples_test)
    WT_pred = WT_pred[Tstart:Tstart+T] * WT_individual  # scale
    WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    # generate training and testing scenarios
    WT_error_scenarios_train = WT_error_scenarios[:N_samples_train]
    WT_error_scenarios_test = WT_error_scenarios[N_samples_train:]

    # perform SUC
    prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha, wind_curtailment = solve_PD(
        params, num_branch, load_bus_all, PTDF, gen_cap_individual,
        gen_pmin_individual, WT_pred, WT_error_scenarios_train,
        P_line_limit, gen_bus_list, WT_bus_list, rng,
        gen_cost, gen_cost_quadra
    )


    # Check the status of the solution first
    if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
        print(f"Solver status: {prob.status}")
        raise ValueError('The problem does not have a feasible solution.')

    # Extract variable values
    gen_power_all = gen_power_all.X
    gen_alpha_all = gen_alpha_all.X
    storage_p = storage_p.X
    storage_soc = storage_soc.X
    storage_alpha = storage_alpha.X
    wind_curtailment = wind_curtailment.X

    # Check power balance with tolerance
    tolerance = 1e-3
    balanced = True
    for t in range(T):
        gen_total = gen_power_all[t, :].sum()
        wind_total = WT_pred[t, :].sum()
        load_total = load_bus_all[t, :].sum()
        wind_curtailment_total = wind_curtailment[t, :].sum()
        # Formula: gen + wind = load + storage (storage positive when charging)
        balance = gen_total + wind_total - wind_curtailment_total - load_total - storage_p[t]
        if abs(balance) > tolerance:
            balanced = False
            print(f"Power balance violation at t={t}:")
            print(f"  Gen total: {gen_total:.6f}")
            print(f"  Wind total: {wind_total:.6f}")
            print(f"  Load total: {load_total:.6f}")
            print(f"  Storage: {storage_p[t]:.6f} (positive=charging)")
            print(f"  Balance error: {balance:.6f}")
            print(f"  Expected: gen+wind = load+storage")
            print(f"  Left side: {gen_total + wind_total:.6f}")
            print(f"  Right side: {load_total + storage_p[t]:.6f}")
    
    if not balanced:
        raise ValueError(f"Power balance not satisfied (tolerance {tolerance})")
    
    t_solve = prob.Runtime

    # test JCC satisfaction rate
    storage_bus_list = [gen_bus_list[0]]
    WT_error_scenario = WT_error_scenarios_test[1]  # use the first scenario to test the actual dispatch and JCC satisfaction
    gen_power_actual, gen_power_total_actual, storage_p_actual, storage_soc_actual, violation_info, fuel_cost_hourly, wind_curtailment_hourly, load_shedding_hourly = solve_PD_actual(gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha,
                      T, WT_error_scenario, num_gen, gen_cap_individual, gen_pmin_individual,
                      storage_capacity, storage_power, storage_efficiency=0.95,
                      load_bus_all=load_bus_all, WT_pred=WT_pred,
                      gen_cost=gen_cost, gen_cost_quadra=gen_cost_quadra)
    # plot_power_balance(load_bus_all, WT_error_scenario + WT_pred, gen_power_actual, storage_p_actual,
    #                    T, scenario_idx=0, save_dir='figure/test', save_name='power_balance')
    # satisfied_rate = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
    #           gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list,
    #           storage_p, storage_alpha, storage_bus_list)

    # Print results summary
    print('------------------------------------')
    print(f'{network_name}, {num_gen} generators, {T}-step horizon')
    print(f'Risk level {epsilon}, radius {theta}, N_WDR {N_WDR}')
    print(f'Storage: {storage_capacity} MWh capacity, ±{storage_power} MW power')
    print('')
    # print(f'the objective value is {prob.objVal}, the out-of-sample JCC rate is {satisfied_rate*100}%')
    print(f'The method used is {method}')
    print(f'The computing time for solving the dispatch is {t_solve} seconds')
    print(f'Storage energy used: {storage_p.sum():.2f} MWh (discharge positive, charge negative)')
    print(f'Storage SOC range: [{storage_soc.min():.2f}, {storage_soc.max():.2f}] MWh')
    print(f'Storage alpha range: [{storage_alpha.min():.4f}, {storage_alpha.max():.4f}]')
    print('')
    # 计算并打印燃油花销、弃风、削负荷
    if fuel_cost_hourly is not None:
        total_fuel_cost = fuel_cost_hourly.sum()
        print(f'Total fuel cost: {total_fuel_cost:.2f} USD')
        print(f'Hourly fuel cost (USD): {fuel_cost_hourly}')
    else:
        print('Fuel cost not calculated (gen_cost or gen_cost_quadra missing)')
    total_wind_curtailment = wind_curtailment_hourly.sum()
    wind_cur_rate = total_wind_curtailment / (WT_pred.sum() + WT_error_scenario.sum())
    total_load_shedding = load_shedding_hourly.sum()
    load_cur_rate = total_load_shedding / load_bus_all.sum()
    print(f'Total wind curtailment: {total_wind_curtailment:.6f} MW')
    print(f'Total load shedding: {total_load_shedding:.6f} MW')
    print(f'Wind curtailment rate: {wind_cur_rate:.2%}')
    print(f'Load shedding rate: {load_cur_rate:.2%}')
    print('------------------------------------')
    # plot the results
    # plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
    #               WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost, storage_p, storage_soc, storage_alpha)
    return load_bus_all, WT_pred, WT_error_scenario, gen_power_all, wind_curtailment, storage_p_actual, storage_soc_actual, fuel_cost_hourly, wind_curtailment_hourly, load_shedding_hourly

if __name__ == '__main__':
    # Create parameter object
    params = PDParams(
        network_name='case5',
        method='CVAR',
        N_WDR=100,
        epsilon=0.03,
        theta=1.3e-1,
        num_gen=3,
        Tstart=0,
        norm_ord=1,
        T=24,
        load_scaling_factor=1,
        storage_capacity=1000.0,
        storage_power=500.0,
        storage_efficiency=0.95,
        storage_soc_init=0.5
    )

    # Run with storage
    solve_PD_instance(params)