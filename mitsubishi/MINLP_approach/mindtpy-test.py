import pyomo.environ as pyo
model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(1.0,10.0),initialize=5.0)
model.y = pyo.Var(within=pyo.Binary)
model.c1 = pyo.Constraint(expr=(model.x-4.0)**2 - model.x <= 50.0*(1-model.y))
model.c2 = pyo.Constraint(expr=model.x*pyo.log(model.x)+5.0 <= 50.0*(model.y))
model.objective = pyo.Objective(expr=model.x, sense=pyo.minimize)
pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt')

model.objective.display()
model.display()
model.pprint()