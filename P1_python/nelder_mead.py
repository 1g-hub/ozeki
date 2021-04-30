
# coding: utf-8

''' Nelder Mead Method
    * clone by https://github.com/fchollet/nelder-mead.git
    * 参考 : http://bicycle1885.hatenablog.com/entry/2015/03/05/040431
    * 元のプログラム : https://github.com/fchollet/nelder-mead/blob/master/nelder_mead.py
    滑降シンプレックス法、アメーバ法と呼ばれる方法


    ２００行程度ものなので比較的簡単
'''

# import module
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# これをインポートするだけでpltの描画がちょっとキレイにおしゃれになる
import seaborn
import copy
import matplotlib.animation as animation
import scipy

# 描画結果を保存するリスト
graphs =[]

def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
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
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print ('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

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
        # reflection（反射）で最適値へ向かっていくのでこれを描画する
        g = plt.plot(xr, rscore, 'o')
        graphs.append(g)
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
        return (x * x + 3 * x * x - 9 * x + 5)

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