import numpy as np
import pyomo.environ as pyo

def milp_daily_optimization(day,ev_agents, price, limit):
    T = ev_agents[0].T #Number of timestep per day
    N_evs = len(ev_agents)

    model = pyo.ConcreteModel()

    model.N_evs = pyo.RangeSet(N_evs)
    model.I = pyo.RangeSet(T)

    model.P = pyo.Param(model.N_evs, initialize={n: ev_agents[n-1].ev_config.p_max for n in model.N_evs})
    model.price = pyo.Param(model.I, initialize={i: price[i-1] for i in model.I}) # Price at each instant of the episode
    model.limit = pyo.Param(model.I, initialize = {i: limit[i-1] for i in model.I}) # Available power before congestion
    model.M_t = pyo.Param(model.N_evs, initialize = {n: ev_agents[n-1].state.n_charge for n in model.N_evs}) #Number of charge to satisfy the demand
    model.t_a = pyo.Param(model.N_evs, initialize = {n: ev_agents[n-1].state.t_a for n in model.N_evs}) #Arrival Times
    model.t_b= pyo.Param(model.N_evs, initialize = {n: ev_agents[n-1].state.t_b for n in model.N_evs}) #Departure Times
    
    model.x = pyo.Var(model.N_evs, model.I, domain = pyo.Binary) # Binary Charging schedule of agents (1 <=> Charging)

    #Aggregated Linear price centralized cost function
    def total_cost(model):
        return sum(model.x[n,i]*model.price[i] for n in model.N_evs for i in model.I)
    
    model.obj = pyo.Objective(rule = total_cost, sense = pyo.minimize)
    
    # Demand Satisfaction constraint
    def demand_satisfaction_rule(model, n):
        return sum(model.x[n, i] for i in model.I) >= model.M_t[n]

    model.demand_constraint = pyo.Constraint(model.N_evs, rule=demand_satisfaction_rule)

    # Network grid congestion satisfaction constraint
    def congestion_satisfaction_rule(model,i):
        return sum(model.x[n,i]*model.P[n] for n in model.N_evs) <= model.limit[i]

    model.congestive_constraint = pyo.Constraint(model.I, rule = congestion_satisfaction_rule)


    # Availability Constraints
    def availability_constraint_rule_arrival(model, n, i):
        # Only enforce for times before arrival
        if i <= model.t_a[n]:
            return model.x[n, i] == 0
        else:
            return pyo.Constraint.Skip  # no constraint for allowed times


    def availability_constraint_rule_departure(model, n, i):
    # Only enforce for times after departure
        if i > model.t_b[n]:
            return model.x[n, i] == 0
        else:
            return pyo.Constraint.Skip  # no constraint for allowed times

    model.availability_arrival = pyo.Constraint(model.N_evs, model.I, rule=availability_constraint_rule_arrival)
    model.availability_departure = pyo.Constraint(model.N_evs, model.I, rule=availability_constraint_rule_departure)

    

    solver = pyo.SolverFactory("gurobi")
    solver.solve(model, tee = False)
    solver.options["OutPutFlat"] = 0

    p_matrix = np.array([
        [
            ev_agents[n-1].ev_config.p_max * pyo.value(model.x[n, i])
            for n in model.N_evs
        ]
        for i in model.I
    ])

    del solver
    del model
    
    return p_matrix