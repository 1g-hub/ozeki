# -*- coding: utf-8 -*-
# CMA-ESの数世代ごとにネルダーミード法をはさむ
import numpy as np
import itertools
import copy
from deap import base
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import P1

N = P1.N_x  # 問題の次元
NGEN = 5800   # 総ステップ数
lambda_cmaes = 10*N

def nelder_mead(f, x_start, step=0.1, no_improve_thr=1.0e-5, no_improv_break=lambda_cmaes+1, max_iter=5000,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''変数の説明
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]
    fbest_n = []
    x_p = []

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        # break after max_iter
        if max_iter and iters >= max_iter:
            return x_p, fbest_n 
        iters += 1

        # break after no_improv_break iterations with no improvement
        print ('...best so far:', best)
        fbest_n.append(P1.get_fitness(res[0][0]))
        x_p.append(res[0][0])
        x_p = x_p[-lambda_cmaes:] #CMA-ESで使用するxの分だけスライス
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return x_p, fbest_n

        # centroid
        x0 = [0.] * dim
        # worstを除く全ての点の平均を求める
        for tup in res[:-1]:
            # enumerate:インデックス付きでループできる
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        # 三角形をぱたっとひっくり返すイメージ
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        
        # 2番目に大きいスコアより小さく、現在の最小スコア以上
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion（拡大）
        # 現在の最小スコアより小さなスコアを出した場合
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            # 引き伸ばした結果どうなったかで更に最もよいスコアを残す
            # 引き伸ばしたスコアがさらに小さい場合
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        # 反射したスコアが大きくなった場合は拡大の反対、縮小を行う
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        # simplex全体を再定義する
        # 最高点(x1)を残しつつ三角形を小さくする
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def f(x):
    V, F = P1.evaluate_f(x)
    if V < P1.eps[0]:
        V = 0
    return (V, F)

def f_n(x): #ネルダーミード法用の関数
    V, F = P1.evaluate_f(x)
    if V >= P1.eps[0]:
        F = 1.0e7 + V
    return F

toolbox = base.Toolbox()
toolbox.register("evaluate", f)


def main():
    np.random.seed(64)
    
    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=[10.0]*N, sigma=0.05, lambda_=lambda_cmaes)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    # halloffame_array = []
    # C_array = []
    # centroid_array = []
    fbest = []  #np.ndarray((NGEN, 1))    #世代ごとのf(x)のベスト
    vbest = np.ndarray((NGEN, 1))
    # best = np.ndarray((NGEN, N))     #世代ごとのxのベスト

    for gen in range(NGEN):
        #数世代ごとにネルダーミード法を挟む
        if gen%100 == 99:
            x_start = np.array(halloffame[0])
            x_p, fbest_nelder = nelder_mead(f_n, x_start)
            for i in range(lambda_cmaes):
                x_p[i] = creator.Individual(x_p[i]) #これCMA-ESにいれるときめっちゃ大事
            population = x_p
            fbest.extend(fbest_nelder)
        else:
            #新たな世代の個体群を生成
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
        fbest.append(halloffame[0].fitness.values[1]) #V, Fで入力しているときは1
        vbest[gen] = halloffame[0].fitness.values[0]
        # best[gen, :N] = halloffame[0]
        print("{} generation's (bestf, bestv) =({}, {})".format(gen+1, halloffame[0].fitness.values[1], vbest[gen]))

        if (gen+1)%100 == 0:
            x = []
            y = []
            f = [0]*P1.P
            g = [0]*P1.M
            h = [0]*int(P1.Q)

            x = halloffame[0]# best[gen]
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

    #グラフ描画 
    # fbest_list = list(itertools.chain.from_iterable(fbest))
    # fbest_nelder.extend(fbest_list)
    # fbest_all = fbest_nelder 
    y = np.array(fbest)
    x = np.arange(1, len(fbest)+1)
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    plt.plot(x, y)
    plt.yscale('log')
    fig.savefig("img.pdf")

if __name__ == "__main__":
    main()