import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import P1_constants
import numpy as np
#定数
I = P1_constants.I
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
# 決定変数の設定
x_t = np.zeros(N_t*I, dtype="float")
y_t = np.zeros(N_t*I, dtype="int")
x_s = np.zeros(N_s*I, dtype="float")
y_s = np.zeros(N_s*I, dtype="int")
x_g = np.zeros(I, dtype="float")
y_g = np.zeros(I, dtype="int")
x_b = np.zeros(I, dtype="float")
y_b = np.zeros(I, dtype="int")

#係数行列
Q_ts = np.zeros(I, dtype="float")

#決定変数初期化用関数
xx_t = {i:x_t[i] for i in range(len(x_t))}
def init_x_t(model, i):
    return xx_t[i]
yy_t = {i:y_t[i] for i in range(len(y_t))}
def init_x_t(model, i):
    return yy_t[i]
xx_s = {i:x_s[i] for i in range(len(x_s))}
def init_x_s(model, i):
    return xx_s[i]
yy_s = {i:y_s[i] for i in range(len(y_s))}
def init_x_s(model, i):
    return yy_s[i]
xx_g = {i:x_g[i] for i in range(len(x_g))}
def init_x_g(model, i):
    return xx_g[i]
yy_g = {i:y_g[i] for i in range(len(y_g))}
def init_x_g(model, i):
    return yy_g[i]
xx_b = {i:x_b[i] for i in range(len(x_b))}
def init_x_b(model, i):
    return xx_b[i]
yy_b = {i:y_b[i] for i in range(len(y_b))}
def init_x_b(model, i):
    return yy_b[i]

#index list
N_t_index_list = list(range(N_t))
N_s_index_list = list(range(N_s))
I_index_list = list(range(I))

#不等式制約
def computeQ(i, Q_ts_i_minus_1):
    Q_ts_i = 0.0
    for j in N_t_index_list:
        Q_ts_i -= model.x_t[I*j+i]
    for j in N_s_index_list:
        Q_ts_i -= model.x_s[I*j+i] 
    Q_ts_i += Q_ts_i_minus_1 + Q_L[i] + Q_loss
    return Q_ts_i

Q_ts[0] = computeQ(0, Q_ts_init)
for i in range(1,I):
    Q_ts[i] = computeQ(i, Q_ts[i-1])

def const_Eq_b(model):
    for i in I_index_list:
        if i < I-1:
            return (Q_ts_min, Q_ts[i], Q_ts_max1)
        else:
            return (Q_ts_min, Q_ts[i], Q_ts_max2)
model.Eq_b = pyo.ConstraintList()
for i in range(I):
    if i < I-1:
        expr = (Q_ts_min, Q_ts[i], Q_ts_max1)
    else:
        expr = (Q_ts_min, Q_ts[i], Q_ts_max2)
    model.Eq_b.add(expr)

def f_sj(j,i):
    return model.x_s[I*j+i].value / (-a_s[j] * model.x_s[I*j+i].value ** 2 + b_s[j] * model.x_s[I*j+i].value + c_s[j])

model.Eq_c = pyo.ConstraintList()
for i in range(I):
    f_sj_sum = 0.0
    for j in range(N_s):
        f_sj_sum += f_sj(j, i)
    expr = (-10.0, a_gs * model.x_g[i] + a_b * model.x_b[i] - f_sj_sum - S_L[i] - S_rm[i], 10.0)
    model.Eq_c.add(expr)

model.Eq_d = pyo.ConstraintList()
for j in range(N_t):
    for i in range(I):
        expr = (Q_t_min[j]*model.y_t[I*j+i].value, model.x_t[I*j+i].value, Q_t_max[j]*model.y_t[I*j+i].value)
        model.Eq_d.add(expr)

model.Eq_e = pyo.ConstraintList()
for j in range(N_s):
    for i in range(I):
        expr = (Q_s_min[j]*model.y_s[I*j+i].value, model.x_s[I*j+i].value, Q_s_max[j]*model.y_s[I*j+i].value)
        model.Eq_e.add(expr)

model.Eq_f = pyo.ConstraintList()
for i in range(I):
    expr = (E_g_min*model.y_g[i].value, a_ge*model.x_g[i].value, E_g_max*model.y_g[i].value)
    model.Eq_f.add(expr)

model.Eq_g = pyo.ConstraintList()
for i in range(I):
    expr = (S_b_min*model.y_b[i].value, a_b*model.x_b[i].value, S_b_max*model.y_b[i].value)
    model.Eq_g.add(expr)

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

model = pyo.ConcreteModel("MINLP-benchmark")
model.IDX_N_t = range(N_t*I)
model.IDX_N_s = range(N_s*I)
model.IDX = range(I)
model.x_t = pyo.Var(model.IDX_N_t, within=pyo.NonNegativeReals, initialize=0.0) #定義の仕方をなんとかせねば
model.y_t = pyo.Var(model.IDX_N_t, within=pyo.Binary, initialize=0)
model.x_s = pyo.Var(model.IDX_N_s, within=pyo.NonNegativeReals, initialize=0.0)
model.y_s = pyo.Var(model.IDX_N_s, within=pyo.Binary, initialize=0)
model.x_g = pyo.Var(model.IDX, within=pyo.NonNegativeReals, initialize=0.0)
model.y_g = pyo.Var(model.IDX, within=pyo.Binary, initialize=0)
model.x_b = pyo.Var(model.IDX, within=pyo.NonNegativeReals, initialize=0.0)
model.y_b = pyo.Var(model.IDX, within=pyo.Binary, initialize=0)
def ObjRule(model):
    f = 0.0
    for i in range(I):
        f += C_Er[i] * E_r(i) + C_Fr[i] * (model.x_g[i].value + model.x_b[i].value)
    return f

def E_r(i):
    f_t = 0.0
    for j in range(N_t):
        f_t += a_t[j] * model.x_t[I*j+i].value 
    
    return f_t + E_L[i] - a_ge * model.x_g[i].value + E_rm[i]

model.objective = pyo.Objective(rule = ObjRule, sense=pyo.minimize)
opt = SolverFactory('mindtpy')
result = opt.solve(model, strategy ='OA') #mip_solver='glpk', nlp_solver='ipopt'

model.display() #結果表示
#model.objective.display()
#model.pprint()