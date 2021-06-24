# -*- coding: utf-8 -*-
#CMA-ESで最適化
import numpy as np
import itertools

from deap import base
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt
import P1

N = P1.N_x  # 問題の次元
NGEN = 3000   # 総ステップ数
lambda_cmaes = 10*N
cnt_f = 0
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
def f(x):
    cnt = 0
    for i in range(P1.I*P1.N_t):
        if(0 < x[cnt] < P1.Q_t_min[0]):
            x[cnt] += P1.Q_t_min[0]
        cnt += 1
    for i in range(P1.I*P1.N_s):
        if(0 < x[cnt] < P1.Q_s_min[0]):
            x[cnt] += P1.Q_s_min[0]
        cnt += 1
    for i in range(P1.I):
        if(0 < x[cnt] < P1.E_g_min):
            x[cnt] += P1.E_g_min
        cnt += 1
    for i in range(P1.I):
        if(0 < x[cnt] < P1.S_b_min):
            x[cnt] += P1.S_b_min
        cnt += 1
    V, F = P1.evaluate_f(x)
    if V < P1.eps[0]:
        V = 0
    global cnt_f
    F += V*(1.0e3 + 1.0e5/NGEN*cnt_f)
    # F += V*1.0e5
    return (F, )

def f_3(x):
    cnt = 0
    for i in range(P1.I*P1.N_t):
        if(0 < x[cnt] < P1.Q_t_min[0]):
            x[cnt] += P1.Q_t_min[0]
        cnt += 1
    for i in range(P1.I*P1.N_s):
        if(0 < x[cnt] < P1.Q_s_min[0]):
            x[cnt] += P1.Q_s_min[0]
        cnt += 1
    for i in range(P1.I):
        if(0 < x[cnt] < P1.E_g_min):
            x[cnt] += P1.E_g_min
        cnt += 1
    for i in range(P1.I):
        if(0 < x[cnt] < P1.S_b_min):
            x[cnt] += P1.S_b_min
        cnt += 1
    V, F = P1.evaluate_f(x)
    if V < P1.eps[0]:
        V = 0
    return (V, F)

def f_2(x):
    V, F = P1.evaluate_f(x)
    if 100 <= V:
        F = 1.0e10
    elif P1.eps[0] <= V < 100 and F < 4100000:
        F = 4100000
    elif V < P1.eps[0] and F< 4100000:
        V = 0
    return (F, V)

toolbox = base.Toolbox()
toolbox.register("evaluate", f)


def main():
    np.random.seed(64)

    # The CMA-ES algorithm 
    strategy = cma.Strategy(centroid=[0.]*N, sigma=100, lambda_=lambda_cmaes)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    # halloffame_array = []
    # C_array = []
    # centroid_array = []
    fbest = []  # np.ndarray((NGEN, 1))    #世代ごとのf(x)のベスト
    vbest = []  # np.ndarray((NGEN, 1))
    # best = np.ndarray((NGEN, N))     #世代ごとのxのベスト

    for gen in range(NGEN):
        # 新たな世代の個体群を生成
        population = toolbox.generate() #shape = (1世代の個体数, 120)
        # print(population[0])
        # print(np.shape(population[0])) #shape = (120,)

        # 個体群の評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        # halloffame_array.append(halloffame[0])
        # C_array.append(strategy.C)
        # centroid_array.append(strategy.centroid)
        fbest.append(halloffame[0].fitness.values[0])
        # fbest.append(halloffame[0].fitness.values[1]) #V, Fで入力しているときは1
        # vbest.append(halloffame[0].fitness.values[0]) #0
        # best[gen, :N] = halloffame[0]
        # print("{} generation's (bestf, bestv) =({}, {})".format(gen+1, fbest[gen], vbest[gen])) 
        print("{} generation's (bestf, bestv) =({})".format(gen+1, fbest[gen])) 
        if (gen+1)%100 == 0:
            x = []
            y = []
            f = [0]*P1.P
            g = [0]*P1.M
            h = [0]*int(P1.Q)

            x = halloffame[0]
            for n in range(P1.N_x):
                if x[n] < 1.0e-10:
                    y.append(0.0)
                else:
                    y.append(1.0)
            
            #evaluation
            f, g, h = P1.evaluation(x, y, f, g, h)

            #output
            print(x)
            print(y)
            for p in range(P1.P):
                print("f%d = %.10g " % (p+1, f[p]))

            V = 0.0
            for m in range(P1.M):
                # print("g%d = %.10g" % (m+1, g[m]))
                if g[m] > 0.0:
                    V += g[m]

            for q in range(P1.Q):
                # print("h%d = %.10g" % (q+1, h[q]))
                abs(q)
                V += abs(h[q])

            #check feasibility
            print('Sum of violation = {:.10g}'.format(V))
            print("Tolerance = {:.2g} ".format(P1.eps[0]))
            if P1.checkFeasibility(x, y):
                print("Input solution is feasible.")
            else:
                print("Input solution is infeasible.")   
        global cnt_f
        cnt_f += 1

    y_f = np.array(fbest)
    y_v = np.array(vbest)
    x = np.arange(1, len(fbest)+1)

    fig1 = plt.figure()
    fig1.subplots_adjust(left=0.2)
    plt.plot(x, y_f)
    plt.yscale('log')
    plt.xlabel('generation')
    plt.ylabel('F')
    fig1.savefig("cmaes_F.pdf")

    fig2 = plt.figure()
    fig2.subplots_adjust(left=0.2)
    plt.plot(x, y_v)
    plt.yscale('log')
    plt.xlabel('generation')
    plt.ylabel('V')
    fig2.savefig("cmaes_V.pdf")

if __name__ == "__main__":
    main()