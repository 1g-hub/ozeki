from pyomo.environ import *

v = {'banana':8, 'grape':3, 'apple':6, 'peach':11}
w = {'banana':5, 'grape':6, 'apple':3, 'peach':7}

limit = 20

M = ConcreteModel()

M.ITEMS = Set(initialize=v.keys())

M.x = Var(M.ITEMS, within=Binary)
M.value = Objective(expr=sum(v[i]*M.x[i] for i in M.ITEMS), sense=maximize)
M.weight = Constraint(expr=sum(w[i]*M.x[i] for i in M.ITEMS) <= limit)

opt = SolverFactory("glpk")
result = opt.solve(M, tee=True)
M.display()