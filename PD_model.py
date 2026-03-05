import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

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
    # Multiple generator variables (T, num_gen)
    gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gen_power")
    gen_alpha_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gen_alpha")

    # Storage variables (microgrid always has storage)
    storage_p = prob.addMVar(T, lb=-storage_power, ub=storage_power, name="storage_p")
    storage_alpha = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="storage_alpha")
    storage_soc = prob.addMVar(T+1, lb=0, ub=storage_capacity, name="storage_soc")
    dt = 1.0  # 1 hour time step

    for t in range(T):
        # power balance constraint at time step t
        prob.addConstr(gen_power_all[t, :].sum() + WT_pred[t, :].sum() + storage_p[t] == load_bus_all[t, :].sum())

        # AGC constraints at time step t
        # Generators and storage share the uncertainty response
        prob.addConstr(gen_alpha_all[t, :].sum() + storage_alpha[t] == 1)

        # first-stage power limit for each generator
        prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)
        prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)

        # SOC update constraint
        prob.addConstr(storage_soc[t+1] == storage_soc[t] - storage_p[t] * dt / storage_efficiency)

        # Initial SOC constraint (only at t=0)
        if t == 0:
            prob.addConstr(storage_soc[0] == storage_capacity * storage_soc_init)

    # Add SOC cycling constraint (end SOC = initial SOC)
    prob.addConstr(storage_soc[T] == storage_soc[0])

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

    # Pmax constraints (multiple generators)
    for t, g in t_g_list:
        if method == 'CVAR':
            b_Ax = gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
            prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] - gen_power_all[t, g] >= s - r[N_WDR_indices])
        elif method == 'FICA':
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

    # Pmin constraints (multiple generators)
    for t, g in t_g_list:
        if method == 'CVAR':
            b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
            prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] + gen_power_all[t, g] >= s - r[N_WDR_indices])
        elif method == 'FICA':
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
    for t in range(T):
        if method == 'CVAR':
            b_Ax = storage_alpha[t] * np.ones(num_WT)
            prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
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
    PTDF_load = PTDF
    storage_bus = gen_bus_list[0]
    PTDF_storage = PTDF[:, storage_bus]

    # line max flow constraints
    P_line_max = P_line_limit
    for t, l in t_l_list:
        # greater than norm constraint
        b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) + PTDF_storage[l] * storage_alpha[t] * np.ones(num_WT) - PTDF_wind[l]
        prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - gp.quicksum(r), theta * N_WDR * b_Ax, norm_ord=norm_ord))

        # main constraints
        prob.addConstr(P_line_max[l] - (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                - (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                + PTDF_storage[l] * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                + PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])

    # line min flow constraints
    P_line_min = -P_line_limit
    for t, l in t_l_list:
        b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) - PTDF_storage[l] * storage_alpha[t] * np.ones(num_WT) + PTDF_wind[l]
        prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))

        # main constraints
        prob.addConstr(-P_line_min[l] + (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                + (PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                - PTDF_storage[l] * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                - PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
# ---------------------------------------------------
    # Define the cost (objective) function
    # fuel cost for all generators
    FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2

    # storage cost (wear and tear cost)
    storage_cost_coeff = 10.0  # USD/MWh (battery degradation cost)
    storage_p_abs = prob.addMVar(T, lb=0, ub=storage_power, name="storage_p_abs")
    prob.addConstr(storage_p_abs >= storage_p)
    prob.addConstr(storage_p_abs >= -storage_p)
    SC = storage_cost_coeff * storage_p_abs
    prob.setObjective(FC.sum() + SC.sum(), GRB.MINIMIZE)
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

    return prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha