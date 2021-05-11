# -*- coding: utf-8 -*-
import numpy as np
import itertools

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt
import P1

N = P1.N_x  # 問題の次元
NGEN = 15000   # 総ステップ数

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def f(x):
    V, F = P1.evaluate_f(x)
    if V < P1.eps[0]:
        V = 0
    return (V, F)

toolbox = base.Toolbox()
toolbox.register("evaluate", f)


def main():
    np.random.seed(64)

    # The CMA-ES algorithm 
    strategy = cma.Strategy(centroid=[10.0]*N, sigma=0.05, lambda_=5*N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    # halloffame_array = []
    # C_array = []
    # centroid_array = []
    fbest = np.ndarray((NGEN, 1))    #世代ごとのf(x)のベスト
    vbest = np.ndarray((NGEN, 1))
    best = np.ndarray((NGEN, N))     #世代ごとのxのベスト

    for gen in range(NGEN):
        # 新たな世代の個体群を生成
        population = toolbox.generate()
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
        fbest[gen] = halloffame[0].fitness.values[1] #V, Fで入力しているときは1
        vbest[gen] = halloffame[0].fitness.values[0]
        best[gen, :N] = halloffame[0]
        print("{} generation's (bestf, bestv) =({}, {})".format(gen+1, fbest[gen], vbest[gen])) 
        if (gen+1)%500 == 0:
            x = []
            y = []
            f = [0]*P1.P
            g = [0]*P1.M
            h = [0]*int(P1.Q)

            x = best[gen]
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
    y_f = np.array(list(itertools.chain.from_iterable(fbest)))
    x = np.arange(1, NGEN+1)

    fig = plt.figure()
    plt.plot(x, y_f)
    plt.show()
    fig.savefig("img.pdf")

if __name__ == "__main__":
    main()