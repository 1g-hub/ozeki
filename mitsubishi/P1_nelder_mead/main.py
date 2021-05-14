
# coding: utf-8

''' Nelder Mead Method
    * clone by https://github.com/fchollet/nelder-mead.git
    * 参考 : http://bicycle1885.hatenablog.com/entry/2015/03/05/040431
    * 元のプログラム : https://github.com/fchollet/nelder-mead/blob/master/nelder_mead.py
    滑降シンプレックス法、アメーバ法と呼ばれる方法


    ２００行程度ものなので比較的簡単
'''

# import module
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# これをインポートするだけでpltの描画がちょっとキレイにおしゃれになる
import seaborn
import copy
import matplotlib.animation as animation
import scipy

import P1

# 描画結果を保存するリスト
graphs =[]

def nelder_mead(f, x_start, step=1., no_improve_thr=1.0e-11, no_improv_break=2000, max_iter=0,
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
    vbest = []
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
        ff, vv = P1.evaluate_f(res[0][0])
        fbest_n.append(ff)
        vbest.append(vv)
        x_p.append(res[0][0])
        x_p = x_p[-100:] #適当
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

# テスト用、最小値を探す関数
def f(x):
    V, F = P1.evaluate_f(x)
    if V >= P1.eps[0]:
        F = 1.0e7 + V
    return F

a = 0.
b = 1.
x = (b - a) * np.random.rand(P1.N_x) + a #初期値として0以上1未満の一様乱数(実数)

#nelder mead
x_p, fbest_nelder, vbest = nelder_mead(f, x)
x_f = x_p[-1]
y = []
f = [0]*P1.P
g = [0]*P1.M
h = [0]*int(P1.Q)
for n in range(P1.N_x):
    if x_f[n] < 1.0e-10:
        y.append(0.0)
    else:
        y.append(1.0)

#evaluation
f, g, h = P1.evaluation(x_f, y, f, g, h)

#output
print(x_f)
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
if P1.checkFeasibility(x_f, y):
    print("Input solution is feasible.")
else:
    print("Input solution is infeasible.")    

y = np.array(fbest_nelder)
x = np.arange(1, len(fbest_nelder)+1)
fig = plt.figure()
fig.subplots_adjust(left=0.2)
plt.plot(x, y)
plt.yscale('log')
fig.savefig("nelder.pdf")


'''
    # 描画する範囲
    x = np.arange(-8, 8, 0.01)
    # ウィンドウの生成
    fig = plt.figure()
    # 関数の描画
    plt.plot(x, f(x))
    # 関数と初期値を与える：変数の次元を一致させる
    print(nelder_mead(f, np.array([-6.])))
    # gifの作成
    # ani = animation.ArtistAnimation(fig, graphs, interval=100)
    # 最適値をプロットする
    # plt.plot(ans[0], f(ans[0]), 'ro')
    # plt.show()
    # ani.save('nelder_mead_method_animation.gif', writer='pillow') #writerをimagebackからpillowに変更 20210428

    # scipyの機能としてある
    # ans = scipy.optimize.minimize(f, np.array([-7.]),method='Nelder-Mead', )
'''