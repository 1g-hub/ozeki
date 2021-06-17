import pyomo.environ as pyo
import P1_constants
import numpy as np
#定数
# I = P1_constants.I
I = 2
N_t = P1_constants.N_t
N_s = P1_constants.N_s
a_ge = P1_constants.a_ge
a_gs = P1_constants.a_gs
a_b = P1_constants.a_b
E_g_min = P1_constants.E_g_min
E_g_max = P1_constants.E_g_max
S_b_min = P1_constants.S_b_min
S_b_max = P1_constants.S_b_max
Q_ts_min = P1_constants.Q_ts_min
Q_ts_max1 = P1_constants.Q_ts_max1 #Q_ts_max1と2の違いは？
Q_ts_max2 = P1_constants.Q_ts_max2
Q_ts_init = P1_constants.Q_ts_init
Q_loss = P1_constants.Q_loss
L_g = P1_constants.L_g
L_b = P1_constants.L_b

#入力データ
def openfile(filename):
    a = []
    with open(filename,mode="r") as f:
        lines = f.read().splitlines()
    a=[float(n) for n in lines]
    return a
def openintfile(filename):
    a = []
    with open(filename,mode="r") as f:
        lines = f.read().splitlines()
    a=[int(n) for n in lines]
    return a

C_Er = openfile("P1_C_Er.conf")
a_t = openfile("P1_a_t.conf")
a_s = openfile("P1_a_s.conf")
b_s = openfile("P1_b_s.conf")
c_s = openfile("P1_c_s.conf")
Q_t_min = openfile("P1_Q_t_min.conf")
Q_t_max = openfile("P1_Q_t_max.conf")
Q_s_min = openfile("P1_Q_s_min.conf")
Q_s_max = openfile("P1_Q_s_max.conf")
L_t = openintfile("P1_L_t.conf")
L_s = openintfile("P1_L_s.conf")
C_Er = openfile("P1_C_Er.conf")
C_Fr = openfile("P1_C_Fr.conf")
E_L = openfile("P1_E_L.conf")
Q_L = openfile("P1_Q_L.conf")
S_L = openfile("P1_S_L.conf")
E_rm = openfile("P1_E_rm.conf")
S_rm = openfile("P1_S_rm.conf")
eps = openfile("P1_tolerance.conf")
model = pyo.ConcreteModel()
# model.x_t = pyo.Var(range(N_t*I), within=pyo.NonNegativeReals, initialize=0.0) #定義の仕方をなんとかせねば
x_t = (Q_t_min[0] + Q_t_max[0])/2
model.y_t = pyo.Var(range(N_t*I), within=pyo.Binary, initialize=0)
# model.x_s = pyo.Var(range(N_s*I), within=pyo.NonNegativeReals, initialize=0.0)
x_s = (Q_s_min[0] + Q_s_max[0])/2
model.y_s = pyo.Var(range(N_s*I), within=pyo.Binary, initialize=0)
# model.x_g = pyo.Var(range(I), within=pyo.NonNegativeReals, initialize=0.0)
x_g = (E_g_min + E_g_max)/2
model.y_g = pyo.Var(range(I), within=pyo.Binary, initialize=0)
# model.x_b = pyo.Var(range(I), within=pyo.NonNegativeReals, initialize=0.0)
x_b = (S_b_min + S_b_max)/2
model.y_b = pyo.Var(range(I), within=pyo.Binary, initialize=0)
Q_ts = [0]*I #model.Q_tsにしなくていいか？
model.display()
def computeQ(i, Q_ts_i_minus_1):
    Q_ts_i = 0.0
    for j in range(N_t):
        Q_ts_i -= x_t*model.y_t[I*j+i].value
    for j in range(N_s):
        Q_ts_i -= x_s*model.y_s[I*j+i].value 
    Q_ts_i += Q_ts_i_minus_1 + Q_L[i] + Q_loss
    return Q_ts_i

Q_ts[0] = computeQ(0, Q_ts_init)
for i in range(1,I):
    Q_ts[i] = computeQ(i, Q_ts[i-1])

model.Eq_b = pyo.ConstraintList()
for i in range(I):
    if i < I-1:
        expr = (Q_ts_min, Q_ts[i], Q_ts_max1)
    else:
        expr = (Q_ts_min, Q_ts[i], Q_ts_max2)
    model.Eq_b.add(expr)

def f_sj(j,i):
    return x_s*model.y_s[I*j+i].value / (-a_s[j] * (x_s*model.y_s[I*j+i].value) ** 2 + b_s[j] * x_s*model.y_s[I*j+i].value + c_s[j])

model.Eq_c = pyo.ConstraintList()
for i in range(I):
    f_sj_sum = 0.0
    for j in range(N_s):
        f_sj_sum += f_sj(j, i)
    expr = a_gs * x_g*model.y_g[i] + a_b * x_b*model.y_b[i] - f_sj_sum - S_L[i] == S_rm[i]
    model.Eq_c.add(expr)

# model.Eq_d = pyo.ConstraintList()
# for j in range(N_t):
#     for i in range(I):
#         expr = (Q_t_min[j]*model.y_t[I*j+i].value, model.x_t[I*j+i].value, Q_t_max[j]*model.y_t[I*j+i].value)
#         model.Eq_d.add(expr)

# model.Eq_e = pyo.ConstraintList()
# for j in range(N_s):
#     for i in range(I):
#         expr = (Q_s_min[j]*model.y_s[I*j+i].value, model.x_s[I*j+i].value, Q_s_max[j]*model.y_s[I*j+i].value)
#         model.Eq_e.add(expr)

# model.Eq_f = pyo.ConstraintList()
# for i in range(I):
#     expr = (E_g_min*model.y_g[i].value, a_ge*model.x_g[i].value, E_g_max*model.y_g[i].value)
#     model.Eq_f.add(expr)

# model.Eq_g = pyo.ConstraintList()
# for i in range(I):
#     expr = (S_b_min*model.y_b[i].value, a_b*model.x_b[i].value, S_b_max*model.y_b[i].value)
#     model.Eq_g.add(expr)

model.Eq_h = pyo.ConstraintList()
for j in range(N_t):
    for i in range(I-1):
        for tau in range(i+2, min(i + L_t[j] + 1 ,I)):
            expr = (model.y_t[I*j+i+1] - model.y_t[I*j+i])*(model.y_t[I*j+i+1] - model.y_t[I*j+tau]) == 0
            model.Eq_h.add(expr)

model.Eq_i = pyo.ConstraintList()
for j in range(N_s):
    for i in range(I-1):
        for tau in range(i+2, min(i + L_s[j] + 1 ,I)):
            expr = (model.y_s[I*j+i+1] - model.y_s[I*j+i])*(model.y_s[I*j+i+1] - model.y_s[I*j+tau]) == 0
            model.Eq_i.add(expr)

model.Eq_j = pyo.ConstraintList()
for i in range(I-1):
    for tau in range(i+2, min(i + L_g + 1 ,I)):
        expr = (model.y_g[i+1] - model.y_g[i])*(model.y_g[i+1] - model.y_g[tau]) == 0
        model.Eq_j.add(expr)

model.Eq_k = pyo.ConstraintList()
for i in range(I-1):
    for tau in range(i+2, min(i + L_b + 1, I)):
        expr = (model.y_b[i+1] - model.y_b[i])*(model.y_b[i+1] - model.y_b[tau]) == 0
        model.Eq_k.add(expr)

def ObjRule(model):
    f = 0.0
    for i in range(I):
        f += C_Er[i] * E_r(i) + C_Fr[i] * (x_g*model.y_g[i].value + x_b*model.y_b[i].value)
    return f

def E_r(i):
    f_t = 0.0
    for j in range(N_t):
        f_t += a_t[j] * x_t*model.y_t[I*j+i].value 
    
    return f_t + E_L[i] - a_ge * x_g*model.y_g[i].value + E_rm[i]

model.objective = pyo.Objective(rule = ObjRule, sense = pyo.minimize)
pyo.SolverFactory('mindtpy').solve(model, strategy ='OA') #mip_solver='glpk', nlp_solver='ipopt'

model.objective.display()
model.display()
model.pprint()