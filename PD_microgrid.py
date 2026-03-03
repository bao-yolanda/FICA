import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
import os
import gurobipy as gp
from gurobipy import GRB
from WT_error_gen import WT_sce_gen
from scipy.linalg import norm
import time
# import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
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
              P_line_limit, gen_bus_list, WT_bus_list):
    # WT_error_scenarios_test has the shape of (N_samples_test, T, num_WT)

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
            L_res.append(line_flow <= P_line_limit[l])
            L_res.append(line_flow >= -P_line_limit[l])

    res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))
    return satisfied_rate

def dual_norm_constr(prob, lhs, rhs, norm_ord=2):
    # this is for lhs >= ||rhs||_norm*
    if norm_ord == 1:
        # return inf-norm
        return [lhs >= rhs, lhs >= -rhs]
    elif norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= rhs @ rhs, lhs_anc == lhs]
    elif norm_ord == np.inf:
        # return 1-norm
        rhs_anc = prob.addMVar(rhs.shape, lb=0, ub=GRB.INFINITY)
        return [lhs >= rhs_anc.sum(), rhs_anc >= rhs, rhs_anc >= -rhs]
    
def dual_norm_constr_exact_method(prob, lhs, rhs_list, norm_ord=2):
    # this is for lhs >= ||rhs||_norm*
    # only implemented the 2-norm
    if norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= gp.quicksum([exp @ exp for exp in rhs_list]), lhs_anc == lhs]
    elif norm_ord == 1:
        # return inf-norm
        return [lhs >= rhs for rhs in rhs_list] + [lhs >= -rhs for rhs in rhs_list]
    else:
        raise NotImplementedError(f'Only 2-norm is implemented, but got {norm_ord}.')

def solve_PD(T, num_gen, num_WT, num_branch, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_train,
              P_line_limit, gen_bus_list, WT_bus_list, N_WDR, epsilon, theta, MIPGap, rng, bigM,
              gen_cost, gen_cost_quadra, gurobi_seed, method="FICA",
              njobs = 1, log_file_name = None, thread = 16, norm_ord = 2,
              storage_capacity=0, storage_power=0, storage_efficiency=0.95, storage_soc_init=0.5):
    '''
    solve the stochastic unit commitment problem with energy storage
    
    Args:
        storage_capacity: Total energy capacity of storage (MWh)
        storage_power: Max charge/discharge power (MW)
        storage_efficiency: Round-trip efficiency (default 0.95)
        storage_soc_init: Initial SOC (default 0.5)
    '''

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0

    t_start = time.time()
    prob = gp.Model('ED')
    gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the scheduled power output of thermal generators
    gen_alpha_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the AGC factors
    
    # Storage variables
    if storage_capacity > 0 and storage_power > 0:
        storage_p = prob.addMVar(T, lb=-storage_power, ub=storage_power, name="storage_p")  # power (positive = discharge, negative = charge)
        storage_alpha = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="storage_alpha")  # storage AGC factor for uncertainty response
        storage_soc = prob.addMVar(T+1, lb=0, ub=storage_capacity, name="storage_soc")  # state of charge
        dt = 1.0  # 1 hour time step

    for t in range(T):
        # power balance constraint at time step t
        if storage_capacity > 0 and storage_power > 0:
            prob.addConstr(gen_power_all[t, :].sum() + WT_pred[t, :].sum() + storage_p[t] == load_bus_all[t, :].sum())
        else:
            prob.addConstr(gen_power_all[t, :].sum() + WT_pred[t, :].sum() == load_bus_all[t, :].sum())

        # AGC constraints at time step t
        if storage_capacity > 0 and storage_power > 0:
            # Generator and storage share the uncertainty response
            prob.addConstr(gen_alpha_all[t, :].sum() + storage_alpha[t] == 1)
        else:
            prob.addConstr(gen_alpha_all[t, :].sum() == 1)

        # first-stage power limit
        # pmax constraint
        prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)
        # pmin constraint
        prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)
        
        # Storage constraints
        if storage_capacity > 0 and storage_power > 0:

            prob.addConstr(storage_soc[t+1] == storage_soc[t] - storage_p[t] * dt / storage_efficiency)
            prob.addConstr(storage_soc[T] == storage_capacity * storage_soc_init)

            if t == 0:
                prob.addConstr(storage_soc[0] == storage_capacity * storage_soc_init)

    # ---------------------------------------------------
    # joint chance constraint

    s = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
    r = prob.addMVar(N_WDR, lb=0, ub=GRB.INFINITY)

    # prepare \xi
    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N_WDR, replace=False)
    # WT_error_scenarios_train has the shape of (N_samples_train, T, num_WT)
    WT_error_scenarios_train = WT_error_scenarios_train[random_var_scenario_index, :, :]
    N_WDR_indices = np.arange(N_WDR)


    t_g_list = [(t, g) for t in range(T) for g in range(num_gen)]

    # Pmax constraints
    for t, g in t_g_list:
        # main constraints
        if method == 'CVAR':
            # greater than norm constraint
            b_Ax = gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
            prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] - gen_power_all[t, g] >= s - r[N_WDR_indices])
        elif method == 'FICA':
            # greater than norm constraint
            b_Ax = gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
            k = np.floor(N_WDR * epsilon).astype(int)
            random_elements = WT_error_scenarios_train.sum(axis=-1)[:,t]
            q_p_plus_base = np.sort(random_elements)[k]
            q_p_minus_base = np.sort(random_elements)[N_WDR-k-1]
            N_p_plus = np.where(random_elements < q_p_plus_base)[0]
            N_p_minus = np.where(random_elements > q_p_minus_base)[0]

            prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_p_plus,t] - gen_power_all[t, g] >= s - r[N_p_plus])
            prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_p_minus,t] - gen_power_all[t, g] >= s - r[N_p_minus])
            prob.addConstr(q_p_plus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)
            prob.addConstr(q_p_minus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)


    # Pmin constraints
    for t, g in t_g_list:
        # main constraints
        if method == 'CVAR':
            # greater than norm constraint
            b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
            prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] + gen_power_all[t, g] >= s - r[N_WDR_indices])
        elif method == 'FICA':
            # greater than norm constraint
            b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))

            k = np.floor(N_WDR * epsilon).astype(int)
            random_elements = WT_error_scenarios_train.sum(axis=-1)[:,t]
            q_p_plus_base = np.sort(random_elements)[k]
            q_p_minus_base = np.sort(random_elements)[N_WDR-k-1]
            N_p_plus = np.where(random_elements < q_p_plus_base)[0]
            N_p_minus = np.where(random_elements > q_p_minus_base)[0]

            prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_p_plus,t] + gen_power_all[t, g] >= s - r[N_p_plus])
            prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_p_minus,t] + gen_power_all[t, g] >= s - r[N_p_minus])
            prob.addConstr(-q_p_plus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)
            prob.addConstr(-q_p_minus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)

    # Storage power constraints for uncertainty response
    if storage_capacity > 0 and storage_power > 0:
        for t in range(T):
            # Storage power max constraint with uncertainty response
            # When WT error is positive (excess generation), storage should be able to absorb more (charge more, negative power)
            # When WT error is negative (generation deficit), storage should be able to discharge more (positive power)
            if method == 'CVAR':
                b_Ax = storage_alpha[t] * np.ones(num_WT)
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                # Storage power constraint: storage_p + storage_alpha * WT_error <= storage_power
                prob.addConstr(storage_power - storage_p[t] - storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] >= s - r[N_WDR_indices])
                prob.addConstr(storage_power + storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] >= s - r[N_WDR_indices])

            elif method == 'FICA':
                b_Ax = storage_alpha[t] * np.ones(num_WT)
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                k = np.floor(N_WDR * epsilon).astype(int)
                random_elements = WT_error_scenarios_train.sum(axis=-1)[:,t]
                q_p_plus_base = np.sort(random_elements)[k]
                q_p_minus_base = np.sort(random_elements)[N_WDR-k-1]
                N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                N_p_minus = np.where(random_elements > q_p_minus_base)[0]

                prob.addConstr(storage_power - storage_p[t] - storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_p_plus,t] >= s - r[N_p_plus])
                prob.addConstr(storage_power - storage_p[t] - storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_p_minus,t] >= s - r[N_p_minus])
                prob.addConstr(storage_power - storage_p[t] - q_p_plus_base * storage_alpha[t] >= s)
                prob.addConstr(storage_power - storage_p[t] - q_p_minus_base * storage_alpha[t] >= s)
                prob.addConstr(storage_power + storage_p[t] + q_p_plus_base * storage_alpha[t] >= s)
                prob.addConstr(storage_power + storage_p[t] + q_p_minus_base * storage_alpha[t] >= s)

    # line flow constraints
    t_l_list = [(t, l) for t in range(T) for l in range(num_branch)]
    PTDF_gen = PTDF[:, gen_bus_list]
    PTDF_wind = PTDF[:, WT_bus_list]
    PTDF_load = PTDF # the load_bus_all is the load at all buses, with shape (T, num_bus)
    # line max flow constraints
    P_line_max = P_line_limit
    for t, l in t_l_list:
        # greater than norm constraint
        if storage_capacity > 0 and storage_power > 0:
            storage_bus = gen_bus_list[0]
            PTDF_storage = PTDF[l, storage_bus]
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) + PTDF_storage * storage_alpha[t] * np.ones(num_WT) - PTDF_wind[l]
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - gp.quicksum(r), theta * N_WDR * b_Ax, norm_ord=norm_ord))

            # main constraints
            prob.addConstr(P_line_max[l] - (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    + PTDF_storage * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
        else:
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) - PTDF_wind[l]
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - gp.quicksum(r), theta * N_WDR * b_Ax, norm_ord=norm_ord))

            # main constraints
            prob.addConstr(P_line_max[l] - (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T) + PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])

    # line min flow constraints
    P_line_min = -P_line_limit
    for t, l in t_l_list:
        if storage_capacity > 0 and storage_power > 0:
            storage_bus = gen_bus_list[0]
            PTDF_storage = PTDF[l, storage_bus]
            b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) - PTDF_storage * storage_alpha[t] * np.ones(num_WT) + PTDF_wind[l]
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
        else:
            b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) + PTDF_wind[l]
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))

        # main constraints
        if storage_capacity > 0 and storage_power > 0:
            storage_bus = gen_bus_list[0]
            PTDF_storage = PTDF[l, storage_bus]
            prob.addConstr(-P_line_min[l] + (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    - PTDF_storage * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
        else:
            prob.addConstr(-P_line_min[l] + (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T) - PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
# ---------------------------------------------------
    # Define the cost (objective) function
    # fuel cost
    FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2
    
    # storage cost (wear and tear cost)
    if storage_capacity > 0 and storage_power > 0:
        storage_cost_coeff = 10.0  # USD/MWh (battery degradation cost)
        # Create auxiliary variable for absolute value of storage power
        storage_p_abs = prob.addMVar(T, lb=0, ub=storage_power, name="storage_p_abs")
        # Add constraints: storage_p_abs >= storage_p and storage_p_abs >= -storage_p
        prob.addConstr(storage_p_abs >= storage_p)
        prob.addConstr(storage_p_abs >= -storage_p)
        SC = storage_cost_coeff * storage_p_abs
        prob.setObjective(FC.sum() + SC.sum(), GRB.MINIMIZE)
    else:
        prob.setObjective(FC.sum(), GRB.MINIMIZE)
    print(f'spent {time.time() - t_start} seconds to build the model.')
    # Solve the problem
    # set MIP gap
    prob.setParam('MIPGap', MIPGap)
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)
    # # fix seed
    prob.setParam('Seed', gurobi_seed)
    prob.setParam('Threads', thread)
    if log_file_name is not None:
        prob.setParam('LogFile', log_file_name)

    # set time limit to 1 hr
    prob.setParam('TimeLimit', 3600)

    # # specify solution method
    # prob.setParam('Method', 5)
    # # disable presolve
    # prob.setParam('NumericFocus', 3)

    prob.optimize()

    if storage_capacity > 0 and storage_power > 0:
        return prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha
    else:
        return prob, gen_power_all, gen_alpha_all

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
    
    # Call solve_PD with or without storage
    if storage_capacity > 0 and storage_power > 0:
        prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha = solve_PD(**input_param_dict)
    else:
        prob, gen_power_all, gen_alpha_all = solve_PD(**input_param_dict)

    # Check the status of the solution first
    if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
        print(f"Solver status: {prob.status}")
        print(f"Solver status description: {prob.StatusToString(prob.status)}")
        raise ValueError('The problem does not have a feasible solution.')

    # Extract variable values
    gen_power_all = gen_power_all.X
    gen_alpha_all = gen_alpha_all.X

    if storage_capacity > 0 and storage_power > 0:
        storage_p = storage_p.X
        storage_soc = storage_soc.X
        storage_alpha = storage_alpha.X
    else:
        storage_p = None
        storage_soc = None
        storage_alpha = None

    t_solve = prob.Runtime

    # test JCC satisfaction rate
    satisfied_rate = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list)

    # Print results summary
    print('------------------------------------')
    print(f'{network_name}, {num_gen} generators, {T}-step horizon')
    print(f'Risk level {epsilon}, radius {theta}, N_WDR {N_WDR}')
    if storage_capacity > 0 and storage_power > 0:
        print(f'Storage: {storage_capacity} MWh capacity, ±{storage_power} MW power')
    print('')
    print(f'the objective value is {prob.objVal}, the out-of-sample JCC rate is {satisfied_rate*100}%')
    print(f'The method used is {method}')
    print(f'The computing time for solving the dispatch is {t_solve} seconds')
    if storage_capacity > 0 and storage_power > 0:
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

def plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost, storage_p=None, storage_soc=None, storage_alpha=None):
    # pick the top 2 generators with the greatest gen_alpha_all.sum(axis=1)
    if num_gen >= 10:
        plot_gen_index = np.argsort(gen_cost)[[num_gen-1, -9]]
    else:
        plot_gen_index = np.argsort(gen_cost)  # plot all generators for small systems
    # make plot for three out-of-sample scenarios
    num_plot_sce = 1
    rng = np.random.RandomState(10)  # fixed random seed for reproducibility
    scenario_set = rng.choice(WT_error_scenarios_test.shape[0], num_plot_sce, replace=False)
    
    # Add storage row if storage is enabled
    num_storage_rows = 3 if storage_p is not None and storage_alpha is not None else (2 if storage_p is not None else 0)
    fig, axs = plt.subplots(len(plot_gen_index)*2+1+num_storage_rows, num_plot_sce, figsize=(8*num_plot_sce, 4 * (len(plot_gen_index)+num_storage_rows)))
    if num_plot_sce <= 1:
        axs = axs[..., None]
    # the first row plots the wind forecast and the actual wind power output
    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        ax_w = axs[0, i]
        # plot the wind forecast
        x = np.arange(T)
        ax_w.step(x, WT_pred.sum(axis=-1), label='forecast')
        # plot the actual wind power output
        ax_w.step(x, WT_pred.sum(axis=-1) + WT_error_scenarios_test.sum(axis=-1)[sce_i], label='actual')
        ax_w.set_title(f'Wind Farm')
        ax_w.set_ylabel('Wind (MW)')
        ax_w.set_xlim(-0.5, T-0.5)
        ax_w.text(1, -0.1, 'Hour', ha='center', va='top', transform=ax_w.transAxes)
        ax_w.legend()
    for i in range(num_plot_sce):
        ax = axs[1::2, i]
        sce_i = scenario_set[i]
        for ig, g in enumerate(plot_gen_index):
            # plot the first-stage power output
            x = np.arange(T)
            ax[ig].step(x, gen_power_all[:, g], label='first-stage')
            # plot the actual power output
            ax[ig].step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * WT_error_scenarios_test[sce_i].sum(axis=-1), label='actual')
            # set x-axis label
            ax[ig].set_title(f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh')
            ax[ig].set_ylabel(f'Gen (MW)')
            # plot Pmin and Pmax as dashed lines
            ax[ig].axhline(gen_pmin_individual[g], color='black', linestyle='--')
            ax[ig].axhline(gen_cap_individual[g], color='black', linestyle='--')
            ax[ig].text(1, -0.1, 'Hour', ha='center', va='top', transform=ax[ig].transAxes)
            ax[ig].set_xlim(-0.5, T-0.5)
            ax[ig].legend()
    # plot alpha values as bars
    for i in range(num_plot_sce):
        ax = axs[2::2, i]
        sce_i = scenario_set[i]
        for ig, g in enumerate(plot_gen_index):
            # plot the alpha values
            x = np.arange(T)
            ax[ig].bar(x, gen_alpha_all[:, g], label='AGC', color='orange')
            ax[ig].set_title(f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh')
            ax[ig].set_ylabel('AGC Factor')
            ax[ig].set_ylim(-1, 1)
            ax[ig].set_xlim(-0.5, T-0.5)
            ax[ig].grid(True, linestyle='--', alpha=0.5)
            # add text hour to the end of the x-axis
            ax[ig].text(1, -0.1, 'Hour', ha='center', va='top', transform=ax[ig].transAxes)
            ax[ig].legend()

    # plot storage if enabled
    if storage_p is not None and storage_soc is not None and storage_alpha is not None:
        for i in range(num_plot_sce):
            # storage power plot
            ax_sp = axs[len(plot_gen_index)*2+1, i]
            x = np.arange(T)
            ax_sp.bar(x, storage_p, label='Storage Power', color='green', alpha=0.7)
            ax_sp.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax_sp.set_title(f'Energy Storage Power')
            ax_sp.set_ylabel('Power (MW)')
            ax_sp.set_xlabel('Hour')
            ax_sp.set_xlim(-0.5, T-0.5)
            ax_sp.grid(True, linestyle='--', alpha=0.5)
            ax_sp.text(1, -0.15, 'Charge (-)/Discharge (+)', ha='center', va='top', transform=ax_sp.transAxes, fontsize=10)
            ax_sp.legend()

            # storage alpha plot
            ax_alpha = axs[len(plot_gen_index)*2+2, i]
            ax_alpha.plot(x, storage_alpha, label='Storage α', color='purple', linewidth=2, marker='o', markersize=4)
            ax_alpha.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax_alpha.set_title(f'Energy Storage AGC Factor (α)')
            ax_alpha.set_ylabel('α')
            ax_alpha.set_xlabel('Hour')
            ax_alpha.set_xlim(-0.5, T-0.5)
            ax_alpha.grid(True, linestyle='--', alpha=0.5)
            ax_alpha.legend()

            # storage SOC plot
            ax_soc = axs[len(plot_gen_index)*2+3, i]
            ax_soc.step(x, storage_soc[:-1], label='SOC', color='blue', linewidth=2)
            ax_soc.fill_between(x, 0, storage_soc[:-1], alpha=0.3, color='blue')
            ax_soc.set_title(f'Energy Storage SOC')
            ax_soc.set_ylabel('SOC (MWh)')
            ax_soc.set_xlabel('Hour')
            ax_soc.set_xlim(-0.5, T-0.5)
            ax_soc.grid(True, linestyle='--', alpha=0.5)
            ax_soc.legend()

    # adjust margins
    plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.03, hspace=0.6, wspace=0.3)
    # save figure to figure/test folder
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    storage_suffix = '_storage' if storage_p is not None else ''
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}{storage_suffix}.pdf')
    plt.savefig(save_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    method = 'CVAR' # FICA, CVAR, and ExactLHS. the method to reformulate the WDRJCC
    N_WDR = 100 # the number of scenarios for the WDRJCC
    epsilon = 0.03 # the risk level. Use 0.06 for plotting Fig. 2
    theta = 1.3e-1 # the Wasserstein radius. Use 2.1e-1 for plotting Fig. 2
    num_gen = 2  # number of generators
    Tstart = 0  # start time index
    norm_ord = 1  # norm order for the WDRJCC
    T = 24  # time horizon in hours
    load_scaling_factor = 1 # the scaling factor for the load
    
    # Storage parameters (set to 0 to disable storage)
    storage_capacity = 50.0  # MWh (total energy capacity)
    storage_power = 20.0    # MW (max charge/discharge power)
    storage_efficiency = 0.95  # round-trip efficiency
    storage_soc_init = 0.5    # initial SOC (50%)
    
    # Run with storage
    solve_PD_instance(num_gen=num_gen, Tstart=Tstart, norm_ord=norm_ord, T=T, method=method, N_WDR=N_WDR, 
                     epsilon=epsilon, theta=theta, load_scaling_factor=load_scaling_factor,
                     storage_capacity=storage_capacity, storage_power=storage_power,
                     storage_efficiency=storage_efficiency, storage_soc_init=storage_soc_init)