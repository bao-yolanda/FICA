import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
import os
from gurobipy import GRB
from WT_error_gen import WT_sce_gen
from scipy.linalg import norm
from PD_model import solve_PD
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from PD_plot import plot_paper
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



def solve_PD_instance(num_gen=38, num_WT=2, Tstart=0, norm_ord=1, T=24, method='FICA', N_WDR=100, epsilon=0.05, theta=1.5e-1, load_scaling_factor=1,
                     storage_capacity=0, storage_power=0, storage_efficiency=0.95, storage_soc_init=0.5):
    N_samples_train = 1000 # the number of wind power scenarios used for training
    N_samples_test = 5000 # the number of wind power scenarios used for testing
    thread = 4
    MIPGap = 0.001
    gurobi_seed = 0

    network_name = 'case5' 
    gen_cap_total_prop = 1 # scale the total generation capacity of the network data

    bigM =1e5 # this is only for "exact"
    log_file_name = None # the log file name
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
    # we then get the load curves at all buses, using the network_load curve
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    ###### set generator capacity
    gen_cap_total = load_total * gen_cap_total_prop  # the total generation capacity
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

    WT_total = 0.6 * load_total
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
    input_param_dict = {'T': T, 'num_gen': num_gen, 'num_WT': num_WT, 'num_branch': num_branch,
                        'load_bus_all': load_bus_all, 'PTDF': PTDF, 'gen_cap_individual': gen_cap_individual,
                        'gen_pmin_individual': gen_pmin_individual, 'WT_pred': WT_pred,
                        'WT_error_scenarios_train': WT_error_scenarios_train, 'P_line_limit': P_line_limit,
                        'gen_bus_list': gen_bus_list, 'WT_bus_list': WT_bus_list, 'N_WDR': N_WDR, 'epsilon': epsilon,
                        'thread': thread,
                        'theta': theta, 'method': method, 'MIPGap': MIPGap, 'gen_cost': gen_cost,
                        'gen_cost_quadra': gen_cost_quadra, 'bigM': bigM, 'gurobi_seed': gurobi_seed,
                        'log_file_name': log_file_name, 'rng': rng, "norm_ord": norm_ord,
                        'storage_capacity': storage_capacity, 'storage_power': storage_power, 
                        'storage_efficiency': storage_efficiency, 'storage_soc_init': storage_soc_init}
    
    # Call solve_PD
    prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha = solve_PD(**input_param_dict)

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

    t_solve = prob.Runtime

    # test JCC satisfaction rate
    storage_bus_list = [gen_bus_list[0]]
    satisfied_rate = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list,
              storage_p, storage_alpha, storage_bus_list)

    # Print results summary
    print('------------------------------------')
    print(f'{network_name}, {num_gen} generators, {T}-step horizon')
    print(f'Risk level {epsilon}, radius {theta}, N_WDR {N_WDR}')
    print(f'Storage: {storage_capacity} MWh capacity, ±{storage_power} MW power')
    print('')
    print(f'the objective value is {prob.objVal}, the out-of-sample JCC rate is {satisfied_rate*100}%')
    print(f'The method used is {method}')
    print(f'The computing time for solving the dispatch is {t_solve} seconds')
    print(f'Storage energy used: {storage_p.sum():.2f} MWh (discharge positive, charge negative)')
    print(f'Storage SOC range: [{storage_soc.min():.2f}, {storage_soc.max():.2f}] MWh')
    print(f'Storage alpha range: [{storage_alpha.min():.4f}, {storage_alpha.max():.4f}]')
    print('')
    print('------------------------------------')
    # plot the results
    plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost, storage_p, storage_soc, storage_alpha)

def plot_all_gen(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost):
    rng = np.random.RandomState(0)  # fixed random seed for reproducibility
    # pick 5 generators to plot, unless there are less than 5 generators
    num_plot_gen = min(5, num_gen)
    # pick random num_plot_gen from 60% generators with the smallest cost, unless there are less than num_plot_gen generators
    top_pick = max(int(0.6 * num_gen), num_plot_gen)
    plot_gen_index = rng.choice(np.argsort(gen_cap_individual)[:top_pick], num_plot_gen, replace=False)
    # make plot for three out-of-sample scenarios
    num_plot_sce = 3
    fig, axs = plt.subplots(num_plot_gen, num_plot_sce, figsize=(5*num_plot_sce, 2 * num_plot_gen))
    for i in range(3):
        ax = axs[:, i]
        for ig, g in enumerate(plot_gen_index):
            # plot the first-stage power output
            x = np.arange(T)
            ax[ig].step(x, gen_power_all[:, g], label='first-stage')
            # plot the actual power output
            ax[ig].step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * WT_error_scenarios_test[i].sum(axis=-1), label='actual')
            # set x-axis label
            ax[ig].set_xlabel('hour')
            # plot Pmin and Pmax as dashed lines
            ax[ig].axhline(gen_pmin_individual[g], color='black', linestyle='--')
            ax[ig].axhline(gen_cap_individual[g], color='black', linestyle='--')
            ax[ig].legend()
            ax[ig].set_title(f'scenario {i}, {method}, generator {g}, eps {epsilon}, theta {theta}')
    plt.tight_layout()
    # save figure to figure/test folder
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}.png')
    plt.savefig(save_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    method = 'CVAR' # FICA, CVAR, and ExactLHS. the method to reformulate the WDRJCC
    N_WDR = 100 # the number of scenarios for the WDRJCC
    epsilon = 0.03 # the risk level. Use 0.06 for plotting Fig. 2
    theta = 1.3e-1 # the Wasserstein radius. Use 2.1e-1 for plotting Fig. 2
    num_gen = 3  # number of generators
    Tstart = 0  # start time index
    norm_ord = 1  # norm order for the WDRJCC
    T = 24  # time horizon in hours
    load_scaling_factor = 1 # the scaling factor for the load

    # Storage parameters (microgrid always has storage)
    storage_capacity = 50.0  # MWh (total energy capacity)
    storage_power = 20.0    # MW (max charge/discharge power)
    storage_efficiency = 0.95  # round-trip efficiency
    storage_soc_init = 0.5    # initial SOC (50%)

    # Run with storage
    solve_PD_instance(num_gen=num_gen, Tstart=Tstart, norm_ord=norm_ord, T=T, method=method, N_WDR=N_WDR,
                     epsilon=epsilon, theta=theta, load_scaling_factor=load_scaling_factor,
                     storage_capacity=storage_capacity, storage_power=storage_power,
                     storage_efficiency=storage_efficiency, storage_soc_init=storage_soc_init)