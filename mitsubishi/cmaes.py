class CMAES:
    def __init__(self, centroid, sigma, lam):
        
        #: 入力次元数
        self.dim = len(centroid)

        #: 世代ごと総個体数λとエリート数μ
        self.lam = lam if lam else int(4 + 3*np.log(self.dim))
        self.mu = int(np.floor(self.lam / 2))

        #: 正規分布中心とその学習率
        self.centroid = np.array(centroid, dtype=np.float64)
        self.c_m = 1.0

        #: 順位重み係数
        weights = [np.log(0.5*(self.lam + 1)) - np.log(i) for i in range(1, 1+self.mu)]
        weights = np.array(weights).reshape(1, -1)
        self.weights = weights / weights.sum()
        self.mu_eff = 1. / (self.weights ** 2).sum()

        #: ステップサイズ： 進化パスpと学習率c
        self.sigma = float(sigma)
        self.p_sigma = np.zeros(self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(
            0, np.sqrt((self.mu_eff - 1)/(self.dim + 1)) - 1
            ) + self.c_sigma

        #: 共分散行列： 進化パスpとrank-μ, rank-one更新の学習率c
        self.C = np.identity(self.dim)
        self.p_c = np.zeros(self.dim)
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2.0 / ((self.dim+1.3)**2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2.0 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff)
            )

    def sample_population(self):
        #: z ~ N(0, I)に従う個体をλ個体生成
        Z = np.random.normal(0, 1, size=(self.lam, self.dim))

        #: Cは対称行列なのでlinalg.eigでもlinalg.eighでも同じ結果
        diagDD, B = np.linalg.eigh(self.C)
        diagD = np.sqrt(diagDD)
        BD = np.matmul(B, np.diag(diagD))

        #: y ~ N(0, C)
        Y = np.matmul(BD, Z.T).T
        #: X ~ N(μ, σC)
        X = self.centroid + self.sigma * Y

        return X

    def update(self, X, fitnesses, gen):
        """ 正規分布パラメータの更新
            X (np.ndarray): 個体群, shape==(self.lam, self.dim)
            fitnesses (list): 適合度
            gen (int): 現在の世代数
        """
        
        """1. Selection and recombination"""
        old_centroid = self.centroid
        old_sigma = self.sigma

        #: fitnessが上位μまでのインデックスを抽出
        elite_indices = np.argsort(fitnesses)[:self.mu]
        
        X_elite = X[elite_indices, :]
        Y_elite = (X_elite - old_centroid) / old_sigma

        X_w = np.matmul(self.weights, X_elite)[0]
        Y_w = np.matmul(self.weights, Y_elite)[0]

        #: 正規分布中心の更新
        self.centroid = (1 - self.c_m) * old_centroid + self.c_m * X_w

        """ 2. Step-size control """
        diagD, B = np.linalg.eigh(self.C)
        diagD = np.sqrt(diagD)
        inv_diagD = 1.0 / diagD

        #: Note. 定義からnp.matmul(B, Z.T).T == np.matmul(C_, Y.T).T
        C_ = np.matmul(np.matmul(B, np.diag(inv_diagD)), B.T)

        new_p_sigma = (1 - self.c_sigma) * self.p_sigma
        new_p_sigma += np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.matmul(C_, Y_w)
        self.p_sigma = new_p_sigma

        E_normal = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim **2)) #:定数パラメータ
        self.sigma = self.sigma * np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.sqrt((self.p_sigma ** 2).sum()) / E_normal - 1)
        )

        """ 3. Covariance matrix adaptation (CMA) """
        #: Note. h_σ(heaviside関数)はステップサイズσが大きいときにはCの更新を中断させる
        left = np.sqrt((self.p_sigma ** 2).sum()) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (gen+1)))
        right = (1.4 + 2 / (self.dim + 1)) * E_normal
        hsigma = 1 if left < right else 0
        d_hsigma = (1 - hsigma) * self.c_c * (2 - self.c_c)

        #: p_cの更新
        new_p_c = (1 - self.c_c) * self.p_c
        new_p_c += hsigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * Y_w
        self.p_c = new_p_c

        #: Cの更新
        new_C = (1 + self.c_1 * d_hsigma - self.c_1 - self.c_mu) * self.C
        new_C += self.c_1 * np.outer(self.p_c, self.p_c)

        #: 愚直な実装（スマートな実装はdeapのcma.pyを参照)
        wyy = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y_i = Y_elite[i]
            wyy += self.weights[0, i] * np.outer(y_i, y_i)
        new_C += self.c_mu * wyy

        self.C = new_C


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse


def levi_func(x1, x2):
    """ https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    theta = np.pi * 3 / 5
    theta = np.pi
    _x1 = x1 * np.cos(theta) - x2 * np.sin(theta)
    _x2 = x1 * np.sin(theta) + x2 * np.cos(theta)

    return 0.1 * (np.sin(3*np.pi*_x1)**2
                  + (_x1 - 1)**2 * (1 + np.sin(3 * np.pi * _x2)**2)
                  + 0.4 * (_x2 - 1)**2 * (1 + np.sin(2*np.pi * _x2)**2))


def main():
    cmaes = CMAES(centroid=[-11, -11], sigma=0.4, lam=12)
    n_generations = 100
    for gen in range(n_generations):

        X = cmaes.sample_population()
        fitnesses = levi_func(X[:, 0], X[:, 1])

        #: パラメータ更新
        cmaes.update(X, fitnesses, gen)