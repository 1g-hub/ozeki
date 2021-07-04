# -*- coding: utf-8 -*-
# ネルダーミード法で最適化を行ってからCMA-ESで最適化
import numpy as np
import random
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
max_iter_n = 1000 # ネルダーミード法の最大ステップ数
nib_n = 200 #ネルダーミード法の変化なし許容回数
NGEN = 5000   # CMA-ESの総ステップ数
lambda_cmaes = 10*N

def nelder_mead(f, x_start, step=0.1, no_improve_thr=1.0e-5, no_improv_break=100, max_iter=1000,
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
    prev_best = f(x_start) #ここでx_startが01になっている
    no_improv = 0
    res = [[x_start, prev_best]]
    fbest_n = []
    Vbest_n = []
    Fbest_n = []
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
            return x_p, fbest_n, Vbest_n, Fbest_n
        iters += 1

        # break after no_improv_break iterations with no improvement
        print (iters, '...best so far:', best)
        # print(res[0][0])
        V_n, f_n = P1.evaluate_f_y(res[0][0])
        fbest_n.append(f_n)
        Vbest_n.append(V_n)
        Fbest_n.append(res[0][1])

        #x_p.append(res[0][0])
        #x_p = x_p[-lambda_cmaes:] #CMA-ESで使用するxの分だけスライス
        x_p = res[0][0]
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return x_p, fbest_n, Vbest_n, Fbest_n

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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
def xc_to_x(x):
    cnt = 0
    X = []
    for i in range(P1.I*P1.N_t):
        if(1 < x[cnt]):
            #x[cnt] += P1.Q_t_min[0]
            X.append(x[cnt] + P1.Q_t_min[0]- 1)
        if(x[cnt] < -1):
            X.append(-x[cnt] + P1.Q_t_min[0] - 1)
        else: X.append(x[cnt])
        cnt += 1
    for i in range(P1.I*P1.N_s):
        if(1 < x[cnt]):
            #x[cnt] += P1.Q_s_min[0]
            X.append(x[cnt] + P1.Q_s_min[0] - 1)
        if(x[cnt] < -1):
            X.append(-x[cnt] + P1.Q_s_min[0] - 1)
        else: X.append(x[cnt])
        cnt += 1
    for i in range(P1.I):
        if(1 < x[cnt]):
            #x[cnt] += P1.E_g_min# /P1.a_ge
            X.append(x[cnt] + P1.E_g_min/P1.a_ge -1)
        if(x[cnt] < -1):
            X.append(-x[cnt] + P1.E_g_min/P1.a_ge -1)
        else: X.append(x[cnt])
        cnt += 1
    for i in range(P1.I):
        if(1 < x[cnt]):
            X.append(x[cnt] + P1.S_b_min/P1.a_b -1)
        if(x[cnt] < -1):
            X.append(-x[cnt] + P1.S_b_min/P1.a_b -1)
        else: X.append(x[cnt])
        cnt += 1
    return X

def f_2(x):
    V, F = P1.evaluate_f(x)
    if V < P1.eps[0]:
        V = 0
    # return (V, F)
    F += V*1.0e12
    return (F, )
def f(x):   #CMA-ESの関数
    # X = xc_to_x(x)
    V, F = P1.evaluate_f(x)
    rho = 1.0e4
    if F < 3999000:
        F += 1.0e10
        V += 1.0e10
    if V < P1.eps[0]:
        V = 0
    F += V*rho
    return (F, )
def f_n(x): #ネルダーミード法用の関数
    V, F = P1.evaluate_f_y(x)
    F += V*1.0e3
    return F
'''def f_n_x(x): #ネルダーミード法用の関数
    V, F = P1.evaluate_f(x)
    F += V*1.0e0
    return F'''

toolbox = base.Toolbox()
toolbox.register("evaluate", f)


def main():
    np.random.seed(19) 
    x_p = []*N
    fbest = []    #世代ごとのf(x)のベスト
    vbest = []
    Fbest = []
    a = 0.99
    b = 1.01
    y_start = (b - a) * np.random.rand(N) + a
    y_p, fbest_nelder, Vbest_nelder, Fbest_nelder = nelder_mead(f_n, y_start, step=1.0, no_improve_thr=1.0e-5, no_improv_break=nib_n, max_iter=max_iter_n) #ネルダーミード法
    y_p = P1.y_01(y_p) #yの値を0か1にする
    x_p = P1.y_to_x(y_p) #yからxの値を代入
        # for i in range(lambda_cmaes):
        #     x_p[i] = creator.Individual(x_p[i])
        # population = x_p
    fbest.extend(fbest_nelder)
    vbest.extend(Vbest_nelder)
    Fbest.extend(Fbest_nelder)
    '''cnt = 0 #x to xc (普通のxをCMAES用のxに変換)
    for i in range(P1.I*P1.N_t):
        if(1 < x_p[cnt]):
            x_p[cnt] -= P1.Q_t_min[0] - 1
        cnt += 1
    for i in range(P1.I*P1.N_s):
        if(1 < x_p[cnt]):
            x_p[cnt] -= P1.Q_s_min[0] - 1
        cnt += 1
    for i in range(P1.I):
        if(1 < x_p[cnt]):
            x_p[cnt] -= P1.E_g_min/P1.a_ge - 1
        cnt += 1
    for i in range(P1.I):
        if(1 < x_p[cnt]):
            x_p[cnt] -= P1.S_b_min/P1.a_b - 1
        cnt += 1'''
    # x_p, fbest_nelder = nelder_mead(f_n_x, np.array(x_p), step=0.1, no_improve_thr=1.0e-5, no_improv_break=100, max_iter=0)   #さらにネルダーミードするなら
    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=x_p, sigma=0.1, lambda_=lambda_cmaes) #平均ベクトルをネルダーミードの最終解にする
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

        # halloffame_array = []
        # C_array = []
        # centroid_array = []
        # best = np.ndarray((NGEN, N))     #世代ごとのxのベスト

    for gen in range(NGEN): #ステップ開始
        #新たな世代の個体群を生成
        population = toolbox.generate()
        if gen == 0:
            population[0] = creator.Individual(x_p) #ネルダーミード法の解を混ぜる
        # 個体群の評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)


        # hall-of-fameの更新
        halloffame.update(population)

        V_c, f_c = P1.evaluate_f(halloffame[0])  
        fbest.append(f_c) #V, Fで入力しているときは1
        vbest.append(V_c)
        bestF = halloffame[0].fitness.values[0]
        Fbest.append(bestF)
        print("{} generation's (bestF, f, V) =({}, {}, {})".format(gen+1, bestF, f_c, V_c))
            # halloffame_array.append(halloffame[0])
            # C_array.append(strategy.C)
            # centroid_array.append(strategy.centroid)

        if (gen+1)%100 == 0:
            y = []
            f = [0]*P1.P
            g = [0]*P1.M
            h = [0]*int(P1.Q)

            x = halloffame[0]# best[gen]
            for n in range(P1.N_x):
                if x[n] < P1.eps[0]:
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
    x = np.arange(1, len(fbest)+1)
    y = np.array(fbest)

    fig = plt.figure(1)
    fig.subplots_adjust(left=0.2)
    plt.yscale('log')
    plt.plot(x, y)
    plt.xlabel("世代", fontname="Noto Serif CJK JP")
    plt.ylabel("$f$")
    fig.savefig("img_f.png")

    x = np.arange(1, len(vbest)+1)
    y = np.array(vbest)

    fig = plt.figure(2)
    fig.subplots_adjust(left=0.2)
    plt.yscale('log')
    plt.plot(x, y)
    plt.xlabel("世代", fontname="Noto Serif CJK JP")
    plt.ylabel("$V$")
    fig.savefig("img_V.png")

    x = np.arange(1, len(Fbest)+1)
    y = np.array(Fbest)

    fig = plt.figure(3)
    fig.subplots_adjust(left=0.2)
    plt.yscale('log')
    plt.plot(x, y)
    plt.xlabel("世代", fontname="Noto Serif CJK JP")
    plt.ylabel("$F$")
    fig.savefig("img_F.png")

if __name__ == "__main__":
    main()