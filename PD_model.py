import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from PD_param import PDParams
from PD_plot import plot_comparison, plot_power_balance, plot_all_gen, plot_paper

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

def solve_PD(params: PDParams, num_branch, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_train,
              P_line_limit, gen_bus_list, WT_bus_list, rng,
              gen_cost, gen_cost_quadra):
    '''
    solve the stochastic unit commitment problem with energy storage

    Args:
        params: PDParams object containing all optimization parameters
        num_branch: Number of transmission lines
        load_bus_all: Load at all buses (T, num_bus)
        PTDF: Power Transfer Distribution Factors
        gen_cap_individual: Individual generator capacity (num_gen,)
        gen_pmin_individual: Individual generator minimum capacity (num_gen,)
        WT_pred: Wind power prediction (T, num_WT)
        WT_error_scenarios_train: Wind error scenarios for training (N_samples_train, T, num_WT)
        P_line_limit: Line flow limits (num_branch,)
        gen_bus_list: Bus indices for generators
        WT_bus_list: Bus indices for wind turbines
        rng: Random number generator
        gen_cost: Generator linear cost coefficients (num_gen,)
        gen_cost_quadra: Generator quadratic cost coefficients (num_gen,)
    '''
    T = params.T
    num_gen = params.num_gen
    num_WT = params.num_WT
    N_WDR = params.N_WDR
    epsilon = params.epsilon
    theta = params.theta
    MIPGap = params.MIPGap
    bigM = params.bigM
    method = params.method
    thread = params.thread
    log_file_name = params.log_file_name
    norm_ord = params.norm_ord
    storage_capacity = params.storage_capacity
    soc_min = storage_capacity *0.2
    storage_power = params.storage_power
    storage_efficiency = params.storage_efficiency
    storage_soc_init = params.storage_soc_init
    storage_cost_coeff = params.storage_cost_coeff
    gurobi_seed = params.gurobi_seed
    
    # Grid connection parameters
    grid_buy_max = getattr(params, 'grid_buy_max', 0.0)
    grid_sell_max = getattr(params, 'grid_sell_max', 0.0)
    grid_bus = getattr(params, 'grid_bus', 0)
    use_tou_pricing = getattr(params, 'use_tou_pricing', False)
    tou_price_file = getattr(params, 'tou_price_file', 'data/tou_price.csv')

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0

    t_start = time.time()
    
    # Load TOU prices if enabled
    if use_tou_pricing:
        try:
            import pandas as pd
            tou_data = pd.read_csv(tou_price_file)
            # Ensure we have T periods
            if len(tou_data) >= T:
                tou_buy_price = tou_data['price_usd_per_mwh'].values[:T]  # buying price from grid
                tou_sell_price = tou_buy_price * 0.7  # selling price to grid (typically lower)
            else:
                print(f"Warning: TOU price file has {len(tou_data)} rows, but T={T}. Using default prices.")
                tou_buy_price = np.ones(T) * 40.0  # default 40 USD/MWh
                tou_sell_price = np.ones(T) * 28.0  # default 28 USD/MWh (70% of buy price)
        except Exception as e:
            print(f"Warning: Failed to load TOU price file {tou_price_file}: {e}. Using default prices.")
            tou_buy_price = np.ones(T) * 40.0
            tou_sell_price = np.ones(T) * 28.0
    else:
        tou_buy_price = np.ones(T) * 40.0  # default flat price
        tou_sell_price = np.ones(T) * 28.0  # default flat selling price
    
    prob = gp.Model('ED')
    # Multiple generator variables (T, num_gen)
    gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gen_power")
    gen_alpha_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gen_alpha")

    # Storage variables (microgrid always has storage)

    storage_p = prob.addMVar(T, lb=-storage_power, ub=storage_power, name="storage_p")
    storage_alpha = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="storage_alpha")
    storage_soc = prob.addMVar(T+1, lb=0, ub=storage_capacity, name="storage_soc")
    # Wind curtailment variables (T, num_WT)
    wind_curtailment = prob.addMVar((T, num_WT), lb=0, ub=WT_pred, name="wind_curtailment")
    
    # Grid connection variables (T,)
    grid_buy = prob.addMVar(T, lb=0, ub=grid_buy_max, name="grid_buy")  # power purchased from grid
    grid_sell = prob.addMVar(T, lb=0, ub=grid_sell_max, name="grid_sell")  # power sold to grid
    
    dt = 1.0  # 1 hour time step

    for t in range(T):
        # power balance constraint at time step t (grid-connected)
        if storage_capacity > 0:
            prob.addConstr(gen_power_all[t, :].sum() + (WT_pred[t, :].sum() - wind_curtailment[t, :].sum()) - storage_p[t] + grid_buy[t] - grid_sell[t] == load_bus_all[t, :].sum())
            # AGC constraints at time step t
            # Generators and storage share the uncertainty response (grid does not participate)
            prob.addConstr(gen_alpha_all[t, :].sum() + storage_alpha[t] == 1)
        else:
            prob.addConstr(gen_power_all[t, :].sum() + (WT_pred[t, :].sum() - wind_curtailment[t, :].sum()) + grid_buy[t] - grid_sell[t] == load_bus_all[t, :].sum())
            # AGC constraints at time step t (no storage)
            prob.addConstr(gen_alpha_all[t, :].sum() == 1)

        # first-stage power limit for each generator
        prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)
        prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)

        # SOC update constraint (only if storage exists)
        if storage_capacity > 0:
            prob.addConstr(storage_soc[t+1] == storage_soc[t] + storage_p[t] * dt / storage_efficiency)

            # Initial SOC constraint (only at t=0)
            if t == 0:
                prob.addConstr(storage_soc[0] == storage_capacity * storage_soc_init)
            if t == T:
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
    if storage_capacity > 0:
        for t in range(T):
            if method == 'CVAR':
                b_Ax = storage_alpha[t] * np.ones(num_WT)
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                prob.addConstr(storage_capacity - storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] >= s - r[N_WDR_indices])

            elif method == 'FICA':
                b_Ax = storage_alpha[t] * np.ones(num_WT)
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                k = np.floor(N_WDR * epsilon).astype(int)
                random_elements = WT_error_scenarios_train.sum(axis=-1)[:,t]
                q_p_plus_base = np.sort(random_elements)[k]
                q_p_minus_base = np.sort(random_elements)[N_WDR-k-1]
                N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                N_p_minus = np.where(random_elements > q_p_minus_base)[0]

                prob.addConstr(storage_capacity - storage_soc[t] - storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_p_plus,t] >= s - r[N_p_plus])
                prob.addConstr(storage_capacity - storage_soc[t] - storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_p_minus,t] >= s - r[N_p_minus])
                prob.addConstr(storage_capacity - storage_soc[t] - storage_p[t] + q_p_plus_base * storage_alpha[t] >= s)
                prob.addConstr(storage_capacity - storage_soc[t] - storage_p[t] + q_p_minus_base * storage_alpha[t] >= s)
   
    # storage constraints lower bounds
    if storage_capacity > 0:
            if method == 'CVAR':
                b_Ax = -storage_alpha[t] * np.ones(num_WT)
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                prob.addConstr(-soc_min + storage_p[t-1] - storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1)[N_WDR_indices,t] >= s - r[N_WDR_indices])




    # # line flow constraints
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
        if storage_capacity > 0:
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) + PTDF_storage[l] * storage_alpha[t] * np.ones(num_WT) - PTDF_wind[l]
        else:
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) - PTDF_wind[l]
        prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - gp.quicksum(r), theta * N_WDR * b_Ax, norm_ord=norm_ord))

        # main constraints
        if storage_capacity > 0:
            prob.addConstr(P_line_max[l] - (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - (PTDF_wind[l] @ (WT_pred[t] - wind_curtailment[t]) + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    + PTDF_storage[l] * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
        else:
            prob.addConstr(P_line_max[l] - (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - (PTDF_wind[l] @ (WT_pred[t] - wind_curtailment[t]) + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    + PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])

    # line min flow constraints
    P_line_min = -P_line_limit
    for t, l in t_l_list:
        if storage_capacity > 0:
            b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) - PTDF_storage[l] * storage_alpha[t] * np.ones(num_WT) + PTDF_wind[l]
        else:
            b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT) + PTDF_wind[l]
        prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))

        # main constraints
        if storage_capacity > 0:
            prob.addConstr(-P_line_min[l] + (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + (PTDF_wind[l] @ (WT_pred[t] - wind_curtailment[t]) + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    - PTDF_storage[l] * (storage_p[t] + storage_alpha[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    - PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
        else:
            prob.addConstr(-P_line_min[l] + (PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * WT_error_scenarios_train.sum(axis=-1, keepdims=True)[N_WDR_indices,t].T)
                                    + (PTDF_wind[l] @ (WT_pred[t] - wind_curtailment[t]) + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T)
                                    - PTDF_load[l] @ load_bus_all[t] >= s - r[N_WDR_indices])
# ---------------------------------------------------
    # Define the cost (objective) function
    # fuel cost for all generators
    FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2
    # wind curtailment cost
    wind_curtailment_cost = getattr(params, 'wind_curtailment_cost', 0.0)
    WC = wind_curtailment_cost * wind_curtailment.sum()

    # grid transaction cost (buying from grid is cost, selling to grid is revenue)
    grid_cost = tou_buy_price @ grid_buy - tou_sell_price @ grid_sell

    # storage cost (wear and tear cost)
    if storage_capacity > 0:
        storage_p_abs = prob.addMVar(T, lb=0, ub=storage_power, name="storage_p_abs")
        prob.addConstr(storage_p_abs >= storage_p)
        prob.addConstr(storage_p_abs >= -storage_p)
        SC = storage_cost_coeff * storage_p_abs
        prob.setObjective(FC.sum() + SC.sum() + WC + grid_cost, GRB.MINIMIZE)
    else:
        prob.setObjective(FC.sum() + WC + grid_cost, GRB.MINIMIZE)
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
    prob.setParam('OutputFlag', 0)  # 禁止求解器输出到控制台
    if log_file_name is not None:
        prob.setParam('LogFile', log_file_name)

    # set time limit to 1 hr
    prob.setParam('TimeLimit', 3600)

    # # specify solution method
    # prob.setParam('Method', 5)
    # # disable presolve
    # prob.setParam('NumericFocus', 3)

    prob.optimize()

    return prob, gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha, wind_curtailment, grid_buy, grid_sell


def solve_PD_actual(gen_power_all, gen_alpha_all, storage_p, storage_soc, storage_alpha,
                      T, WT_error_scenario, num_gen, gen_cap_individual, gen_pmin_individual,
                      storage_capacity, storage_power, storage_efficiency=0.95,
                      grid_buy=None, grid_sell=None,
                      load_bus_all=None, WT_pred=None, gen_cost=None, gen_cost_quadra=None,
                      tou_buy_price=None, tou_sell_price=None):
    """
    实时调度：根据实际风电误差调整功率，并对越限进行限制，确保功率平衡。
    计算燃油花销、弃风、削负荷等指标。

    Args:
        gen_power_all: 计划的发电机功率 (T, num_gen)
        gen_alpha_all: 发电机响应系数 (T, num_gen)
        storage_p: 计划的储能功率 (T,)
        storage_soc: 计划的储能SOC (T+1,)
        storage_alpha: 储能响应系数 (T,)
        WT_error_scenario: 实际风电误差 (T, num_WT)
        storage_capacity: 储能容量
        storage_power: 储能功率限制
        storage_efficiency: 储能效率
        grid_buy: 计划从电网购电功率 (T,)，如果为None，则设为0。
        grid_sell: 计划向电网售电功率 (T,)，如果为None，则设为0。
        load_bus_all: 各母线负荷 (T, num_bus) 或总负荷 (T,)。如果为None，则跳过功率平衡检查。
        WT_pred: 风电预测 (T, num_WT) 或总预测 (T,)。如果为None，则跳过功率平衡检查。
        gen_cost: 发电机线性成本系数 (num_gen,)，用于计算燃油花销。如果为None，则跳过燃油花销计算。
        gen_cost_quadra: 发电机二次成本系数 (num_gen,)，用于计算燃油花销。如果为None，则跳过燃油花销计算。
        tou_buy_price: 分时购电价格 (T,)，如果为None，则使用默认价格。
        tou_sell_price: 分时售电价格 (T,)，如果为None，则使用默认价格。

    Returns:
        gen_power_actual: 实际发电机功率 (T, num_gen)
        gen_power_total_actual: 实际总发电机功率 (T,)
        storage_p_actual: 实际储能功率 (T,)
        storage_soc_actual: 实际储能SOC (T+1,)
        violation_info: 违约信息字典，新增'balance_violations'键记录功率不平衡。
        fuel_cost_hourly: 每小时燃油花销 (T,)，如果gen_cost和gen_cost_quadra提供；否则为None。
        wind_curtailment_hourly: 每小时弃风量 (T,)，正数表示弃风。
        load_shedding_hourly: 每小时削负荷量 (T,)，正数表示削负荷。
        grid_buy_actual: 实际从电网购电功率 (T,)
        grid_sell_actual: 实际向电网售电功率 (T,)
    """

    # 存储所有时刻的实际功率
    gen_power_actual = np.zeros((T, num_gen))
    storage_p_actual = np.zeros(T)
    storage_soc_actual = np.zeros(T+1)
    
    # 处理grid交易（如果未提供则设为0）
    if grid_buy is None:
        grid_buy_actual = np.zeros(T)
    else:
        # 提取变量值（如果是Gurobi变量）
        if hasattr(grid_buy[0], 'X'):
            grid_buy_actual = np.array([gb.X for gb in grid_buy])
        else:
            grid_buy_actual = grid_buy.copy()
    
    if grid_sell is None:
        grid_sell_actual = np.zeros(T)
    else:
        if hasattr(grid_sell[0], 'X'):
            grid_sell_actual = np.array([gs.X for gs in grid_sell])
        else:
            grid_sell_actual = grid_sell.copy()

    # 初始化SOC
    storage_soc_actual[0] = storage_soc[0].X if hasattr(storage_soc[0], 'X') else storage_soc[0]

    # 违约信息
    violation_info = {
        'gen_violations': [],
        'storage_power_violations': [],
        'storage_soc_violations': [],
        'balance_violations': []
    }

    # 计算总负荷和总风电预测（如果提供）
    if load_bus_all is not None:
        if load_bus_all.ndim == 2:
            load_total = load_bus_all.sum(axis=1)  # (T,)
        else:
            load_total = load_bus_all  # 假设已经是(T,)
    else:
        load_total = None
        print("警告：未提供负荷数据，跳过功率平衡检查。")
    
    if WT_pred is not None:
        if WT_pred.ndim == 2:
            WT_pred_total = WT_pred.sum(axis=1)  # (T,)
        else:
            WT_pred_total = WT_pred  # 假设已经是(T,)
    else:
        WT_pred_total = None
        print("警告：未提供风电预测数据，跳过功率平衡检查。")

    # 初始化输出指标
    fuel_cost_hourly = np.zeros(T) if gen_cost is not None and gen_cost_quadra is not None else None
    wind_curtailment_hourly = np.zeros(T)
    WT_actual = np.zeros(T)
    load_shedding_hourly = np.zeros(T)
    tolerance = 1e-3

    for t in range(T):
        # 实际风电误差
        Delta_W_t = np.sum(WT_error_scenario[t, :])  # 总误差
        WT_actual[t] = WT_pred_total[t] + Delta_W_t if WT_pred_total is not None else None
        # 计算初始调整功率（按照AGC分配）
        gen_power_initial = gen_power_all[t, :].X if hasattr(gen_power_all[t, 0], 'X') else gen_power_all[t, :]
        storage_p_initial = storage_p[t].X if hasattr(storage_p[t], 'X') else storage_p[t]

        # 初始调整
        gen_power_adjusted = gen_power_initial + gen_alpha_all[t, :] * Delta_W_t
        storage_p_adjusted = storage_p_initial + storage_alpha[t] * Delta_W_t

        gen_power_actual[t, :] = np.clip(gen_power_adjusted, gen_pmin_individual, gen_cap_individual)
        storage_p_actual[t] = np.clip(storage_p_adjusted, -storage_power, storage_power)

        # 检查修正后的SOC
        storage_soc_next_corrected = storage_soc_actual[t] + storage_p_actual[t] 
        storage_soc_actual[t+1] = np.clip(storage_soc_next_corrected, 0, storage_capacity)

        # 调整储能功率以适应SOC限制
        if storage_soc_actual[t+1] != storage_soc_next_corrected:
            storage_p_actual[t] = (storage_soc_actual[t+1] - storage_soc_actual[t]) * storage_efficiency
        else:
            # 无违约，直接使用调整后的功率
            gen_power_actual[t, :] = gen_power_adjusted
            storage_p_actual[t] = storage_p_adjusted
            storage_soc_actual[t+1] = storage_soc_next_corrected

        # ===== 5. 检查功率平衡（如果数据齐全） =====
        if load_total is not None and WT_pred_total is not None:            
            # 实际总发电机功率
            gen_total_actual = gen_power_actual[t, :].sum()
            # 功率平衡： gen + wind + grid_buy - grid_sell = load + storage (storage正为充电，消耗功率)
            balance = gen_total_actual + WT_actual[t] + grid_buy_actual[t] - grid_sell_actual[t] - load_total[t] - storage_p_actual[t]
            # 计算弃风和削负荷
            if balance > tolerance:
                wind_curtailment_hourly[t] = balance  # 正平衡视为弃风
                load_shedding_hourly[t] = 0.0
            elif balance < -tolerance:
                load_shedding_hourly[t] = -balance    # 负平衡视为削负荷
                wind_curtailment_hourly[t] = 0.0
            else:
                wind_curtailment_hourly[t] = 0.0
                load_shedding_hourly[t] = 0.0
        else:
            wind_curtailment_hourly[t] = 0.0
            load_shedding_hourly[t] = 0.0

        # ===== 6. 计算燃油花销（如果成本系数提供） =====
        if fuel_cost_hourly is not None:
            # 燃油花销 = 线性成本 * 实际功率 + 二次成本 * 实际功率^2
            fuel_cost_hourly[t] = np.sum(gen_cost * gen_power_actual[t, :] + gen_cost_quadra * gen_power_actual[t, :] ** 2)

    # 计算总功率
    gen_power_total_actual = np.sum(gen_power_actual, axis=1)
    # Plot power balance including grid transactions
    if load_total is not None and WT_pred_total is not None:
        plot_power_balance(load_total, WT_actual, WT_pred_total, gen_power_total_actual, storage_p_actual, 
                          load_shedding_hourly, wind_curtailment_hourly, T, save_dir='results', save_name='power_balance',
                          grid_buy=grid_buy_actual, grid_sell=grid_sell_actual,
                          tou_buy_price=tou_buy_price, tou_sell_price=tou_sell_price)

    return gen_power_actual, gen_power_total_actual, storage_p_actual, storage_soc_actual, violation_info, fuel_cost_hourly, wind_curtailment_hourly, load_shedding_hourly, grid_buy_actual, grid_sell_actual 