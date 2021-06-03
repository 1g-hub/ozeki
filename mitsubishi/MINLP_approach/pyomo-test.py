import pyomo.environ as pyo

v = {'banana':8, 'grape':3, 'apple':6, 'peach':11}
w = {'banana':5, 'grape':6, 'apple':3, 'peach':7}

limit = 20

M = pyo.ConcreteModel()

M.ITEMS = pyo.Set(initialize=v.keys())

M.x = pyo.Var(M.ITEMS, within=pyo.Binary)
M.value = pyo.Objective(expr=sum(v[i]*M.x[i] for i in M.ITEMS), sense=pyo.maximize)
M.weight = pyo.Constraint(expr=sum(w[i]*M.x[i] for i in M.ITEMS) <= limit)

opt = pyo.SolverFactory("glpk")
result = opt.solve(M, tee=True)
M.display()