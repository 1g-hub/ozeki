import P1_constants
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
Q_ts_max1 = P1_constants.Q_ts_max1
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

x = []
y = []

x_t = [[0]*I for i in range(N_t)]
y_t = [[0]*I for i in range(N_t)]
x_s = [[0]*I for i in range(N_s)]
y_s = [[0]*I for i in range(N_s)]

x_g = [0]*I
y_g = [0]*I
x_b = [0]*I
y_b = [0]*I
Q_ts = [0]*I

#Constructor
N_x = I * (N_t + N_s + 2)
N_y = I * (N_t + N_s + 2)
P = 1
M = I * (6 + 2 * N_t + 2 * N_s)
Q = (I - L_t[0]) + (I - L_s[0]) + (I - L_s[1]) + (I - L_g) + (I - L_b) + I

f = [0]*P
g = [0]*M
h = [0]*int(Q)

def evaluation(x, y, f, g, h):
    #Substitution of x and y

    cnt = 0
    for n in range(N_t):
        for i in range(I):
            x_t[n][i] = x[cnt]
            y_t[n][i] = y[cnt]
            cnt += 1

    for n in range(N_s):
        for i in range(I):
            x_s[n][i] = x[cnt]
            y_s[n][i] = y[cnt]
            cnt += 1

    for i in range(I):
        x_g[i] = x[cnt]
        y_g[i] = y[cnt]
        cnt += 1

    for i in range(I):
        x_b[i] = x[cnt]
        y_b[i] = y[cnt]
        cnt += 1

    #Compute Q_ts
    Q_ts[0] = computeQ(0, Q_ts_init)
    for i in range(1,I):
        Q_ts[i] = computeQ(i, Q_ts[i-1])

    #Compute objective function - Eq. (6a)
    f[0] = 0.0
    for i in range(I):
        f[0] += C_Er[i] * E_r(i) + C_Fr[i] * (x_g[i] + x_b[i])

    """ Compute constraint conditions """
    #Eq. (6b) and (6c)
    id_g = 0
    for i in range(I):
        g[id_g] = Q_ts_min - Q_ts[i]
        id_g += 1
    for i in range(I):
        if i < I-1:
            g[id_g] = Q_ts[i] - Q_ts_max1
        else:
            g[id_g] = Q_ts[i] - Q_ts_max2
        id_g += 1

    #Eq. (6d)
    id_h = 0
    for i in range(I):
        h[id_h] = a_gs * x_g[i] + a_b * x_b[i]
        for j in range(N_s):
            h[id_h] -= f_sj(j, i)
        h[id_h] -= S_L[i]
        h[id_h] -= S_rm[i]
        id_h += 1

    #Eq. (6e)
    for j in range(N_t):
        for i in range(I):
            g[id_g] = Q_t_min[j] * y_t[j][i] - x_t[j][i]
            id_g += 1

    for j in range(N_t):
        for i in range(I):
            g[id_g] = x_t[j][i] - Q_t_max[j] * y_t[j][i]
            id_g += 1

    #Eq. (6f)
    for j in range(N_s):
        for i in range(I):
            g[id_g] = Q_s_min[j] * y_s[j][i] - x_s[j][i]
            id_g += 1

    for j in range(N_s):
        for i in range(I):
            g[id_g] = x_s[j][i] - Q_s_max[j] * y_s[j][i]
            id_g += 1

    #Eq. (6g)
    for i in range(I):
        g[id_g] = E_g_min * y_g[i] - a_ge * x_g[i]
        id_g += 1
    for i in range(I):
        g[id_g] = a_ge * x_g[i] - E_g_max * y_g[i]
        id_g += 1

    #Eq. (6h)
    for i in range(I):
        g[id_g] = S_b_min * y_b[i] - a_b * x_b[i]
        id_g += 1

    for i in range(I):
        g[id_g] = a_b * x_b[i] - S_b_max * y_b[i]
        id_g += 1

    #Eq. (6i)
    for j in range(N_s):
        for i in range(I-1):
            for tau in range(i+2,i + L_s[j]):
                if tau < I:
                    h[id_h] = (y_s[j][i + 1] - y_s[j][i]) * (y_s[j][i + 1] - y_s[j][tau])
                    id_h += 1

    #Eq. (6k)
    for i in range(I-1):
        for tau in range(i+2,i+L_g+1):
            if tau > I:
                h[id_h] = (y_g[i + 1] - y_g[i]) * (y_g[i + 1] - y_g[tau])
                id_h += 1

    #Eq. (6l)
    for i in range(I-1):
        for tau in range(i+2,i+L_g):
            if tau < I:
                h[id_h] = (y_b[i + 1] - y_b[i]) * (y_b[i + 1] - y_b[tau])
                id_h += 1
    
    return f,g,h

""" Check feasibility """
def checkFeasibility(x,y):
    feasibility = True
    for n in range(N_y):
        if (y[n] < 0 or y[n] > 2):
            feasibility = False
    
    """ Compute f, g, and h """
    #f = [P]
    #g = [M]
    #h = [Q]
    evaluation(x, y, f, g, h)

    """ Check inequality conditions """
    eps = openfile("P1_tolerance.conf")
    for m in range(M):
        if g[m] > eps[0]:
            feasibility = False
    
    """ Check equality conditions """
    for q in range(Q):
        if abs(h[q]) > eps[0]:
            feasibility = False
    
    return feasibility


def E_r(i):
    f_t = 0.0
    for j in range(N_t):
        f_t += a_t[j] * x_t[j][i]
    
    return f_t + E_L[i] - a_ge * x_g[i] + E_rm[i]

def computeQ(i,Q_ts_i_minus_1):
    Q_ts_i = 0.0
    for j in range(N_t):
        Q_ts_i -= x_t[j][i]
    
    for j in range(N_s):
        Q_ts_i -= x_s[j][i]
    Q_ts_i += Q_ts_i_minus_1 + Q_L[i] + Q_loss
    return Q_ts_i

def f_sj(j,i):
    return x_s[j][i] / (-a_s[j] * x_s[j][i] * x_s[j][i] + b_s[j] * x_s[j][i] + c_s[j])


#def get_fitness(population):
    #変数はxのみで目的関数fの評価を行う
    #この関数内でy,f,g,hを定義できるか
def get_fitness(x):
    #Substitution of x and y
    y = []
    for n in range(N_x):
        if x[n] < 1.0E-10:
            y.append(0.0)
        else:
            y.append(1.0)
    f = [0]*P
    g = [0]*M
    h = [0]*int(Q)
    cnt = 0
    for n in range(N_t):
        for i in range(I):
            x_t[n][i] = x[cnt]
            y_t[n][i] = y[cnt]
            cnt += 1

    for n in range(N_s):
        for i in range(I):
            x_s[n][i] = x[cnt]
            y_s[n][i] = y[cnt]
            cnt += 1

    for i in range(I):
        x_g[i] = x[cnt]
        y_g[i] = y[cnt]
        cnt += 1

    for i in range(I):
        x_b[i] = x[cnt]
        y_b[i] = y[cnt]
        cnt += 1

    #Compute Q_ts
    Q_ts[0] = computeQ(0, Q_ts_init)
    for i in range(1,I):
        Q_ts[i] = computeQ(i, Q_ts[i-1])

    #Compute objective function - Eq. (6a)
    f[0] = 0.0
    for i in range(I):
        f[0] += C_Er[i] * E_r(i) + C_Fr[i] * (x_g[i] + x_b[i])

    """ Compute constraint conditions """
    #Eq. (6b) and (6c)
    id_g = 0
    for i in range(I):
        g[id_g] = Q_ts_min - Q_ts[i]
        id_g += 1
    for i in range(I):
        if i < I-1:
            g[id_g] = Q_ts[i] - Q_ts_max1
        else:
            g[id_g] = Q_ts[i] - Q_ts_max2
        id_g += 1

    #Eq. (6d)
    id_h = 0
    for i in range(I):
        h[id_h] = a_gs * x_g[i] + a_b * x_b[i]
        for j in range(N_s):
            h[id_h] -= f_sj(j, i)
        h[id_h] -= S_L[i]
        h[id_h] -= S_rm[i]
        id_h += 1

    #Eq. (6e)
    for j in range(N_t):
        for i in range(I):
            g[id_g] = Q_t_min[j] * y_t[j][i] - x_t[j][i]
            id_g += 1

    for j in range(N_t):
        for i in range(I):
            g[id_g] = x_t[j][i] - Q_t_max[j] * y_t[j][i]
            id_g += 1

    #Eq. (6f)
    for j in range(N_s):
        for i in range(I):
            g[id_g] = Q_s_min[j] * y_s[j][i] - x_s[j][i]
            id_g += 1

    for j in range(N_s):
        for i in range(I):
            g[id_g] = x_s[j][i] - Q_s_max[j] * y_s[j][i]
            id_g += 1

    #Eq. (6g)
    for i in range(I):
        g[id_g] = E_g_min * y_g[i] - a_ge * x_g[i]
        id_g += 1
    for i in range(I):
        g[id_g] = a_ge * x_g[i] - E_g_max * y_g[i]
        id_g += 1

    #Eq. (6h)
    for i in range(I):
        g[id_g] = S_b_min * y_b[i] - a_b * x_b[i]
        id_g += 1

    for i in range(I):
        g[id_g] = a_b * x_b[i] - S_b_max * y_b[i]
        id_g += 1

    #Eq. (6i)
    for j in range(N_s):
        for i in range(I-1):
            for tau in range(i+2,i + L_s[j]):
                if tau < I:
                    h[id_h] = (y_s[j][i + 1] - y_s[j][i]) * (y_s[j][i + 1] - y_s[j][tau])
                    id_h += 1

    #Eq. (6k)
    for i in range(I-1):
        for tau in range(i+2,i+L_g+1):
            if tau > I:
                h[id_h] = (y_g[i + 1] - y_g[i]) * (y_g[i + 1] - y_g[tau])
                id_h += 1

    #Eq. (6l)
    for i in range(I-1):
        for tau in range(i+2,i+L_g):
            if tau < I:
                h[id_h] = (y_b[i + 1] - y_b[i]) * (y_b[i + 1] - y_b[tau])
                id_h += 1
    


    #制約違反の計算
    V = 0.0
    for m in range(M):
        #print("g%d = %.10g" % (m+1, g[m]))
        if g[m] > 0.0:
            V += g[m]
    
    
    for q in range(Q):
        #print("h%d = %.10g" % (q+1, h[q]))
        V += abs(h[q])
    f[0] += V*10**12
    return  f[0]


def evaluate_f(x):
    #Substitution of x and y
    y = []
    for n in range(N_x):
        if x[n] < 1.0E-10:
            y.append(0.0)
        else:
            y.append(1.0)
    f = [0]*P
    g = [0]*M
    h = [0]*int(Q)
    cnt = 0
    for n in range(N_t):
        for i in range(I):
            x_t[n][i] = x[cnt]
            y_t[n][i] = y[cnt]
            cnt += 1

    for n in range(N_s):
        for i in range(I):
            x_s[n][i] = x[cnt]
            y_s[n][i] = y[cnt]
            cnt += 1

    for i in range(I):
        x_g[i] = x[cnt]
        y_g[i] = y[cnt]
        cnt += 1

    for i in range(I):
        x_b[i] = x[cnt]
        y_b[i] = y[cnt]
        cnt += 1

    #Compute Q_ts
    Q_ts[0] = computeQ(0, Q_ts_init)
    for i in range(1,I):
        Q_ts[i] = computeQ(i, Q_ts[i-1])

    #Compute objective function - Eq. (6a)
    f[0] = 0.0
    for i in range(I):
        f[0] += C_Er[i] * E_r(i) + C_Fr[i] * (x_g[i] + x_b[i])

    """ Compute constraint conditions """
    #Eq. (6b) and (6c)
    id_g = 0
    for i in range(I):
        g[id_g] = Q_ts_min - Q_ts[i]
        id_g += 1
    for i in range(I):
        if i < I-1:
            g[id_g] = Q_ts[i] - Q_ts_max1
        else:
            g[id_g] = Q_ts[i] - Q_ts_max2
        id_g += 1

    #Eq. (6d)
    id_h = 0
    for i in range(I):
        h[id_h] = a_gs * x_g[i] + a_b * x_b[i]
        for j in range(N_s):
            h[id_h] -= f_sj(j, i)
        h[id_h] -= S_L[i]
        h[id_h] -= S_rm[i]
        id_h += 1

    #Eq. (6e)
    for j in range(N_t):
        for i in range(I):
            g[id_g] = Q_t_min[j] * y_t[j][i] - x_t[j][i]
            id_g += 1

    for j in range(N_t):
        for i in range(I):
            g[id_g] = x_t[j][i] - Q_t_max[j] * y_t[j][i]
            id_g += 1

    #Eq. (6f)
    for j in range(N_s):
        for i in range(I):
            g[id_g] = Q_s_min[j] * y_s[j][i] - x_s[j][i]
            id_g += 1

    for j in range(N_s):
        for i in range(I):
            g[id_g] = x_s[j][i] - Q_s_max[j] * y_s[j][i]
            id_g += 1

    #Eq. (6g)
    for i in range(I):
        g[id_g] = E_g_min * y_g[i] - a_ge * x_g[i]
        id_g += 1
    for i in range(I):
        g[id_g] = a_ge * x_g[i] - E_g_max * y_g[i]
        id_g += 1

    #Eq. (6h)
    for i in range(I):
        g[id_g] = S_b_min * y_b[i] - a_b * x_b[i]
        id_g += 1

    for i in range(I):
        g[id_g] = a_b * x_b[i] - S_b_max * y_b[i]
        id_g += 1

    #Eq. (6i)
    for j in range(N_s):
        for i in range(I-1):
            for tau in range(i+2,i + L_s[j]):
                if tau < I:
                    h[id_h] = (y_s[j][i + 1] - y_s[j][i]) * (y_s[j][i + 1] - y_s[j][tau])
                    id_h += 1

    #Eq. (6k)
    for i in range(I-1):
        for tau in range(i+2,i+L_g+1):
            if tau > I:
                h[id_h] = (y_g[i + 1] - y_g[i]) * (y_g[i + 1] - y_g[tau])
                id_h += 1

    #Eq. (6l)
    for i in range(I-1):
        for tau in range(i+2,i+L_g):
            if tau < I:
                h[id_h] = (y_b[i + 1] - y_b[i]) * (y_b[i + 1] - y_b[tau])
                id_h += 1
    


    #制約違反の計算
    V = 0.0
    for m in range(M):
        #print("g%d = %.10g" % (m+1, g[m]))
        if g[m] > 0.0:
            V += g[m]
    
    
    for q in range(Q):
        #print("h%d = %.10g" % (q+1, h[q]))
        V += abs(h[q])
    #V *= 10**10
    return  V,f[0]