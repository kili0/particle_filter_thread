# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rd
import pandas as pd
import time

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

# import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from logging import StreamHandler, Formatter, INFO, getLogger

def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger = getLogger()
    logger.addHandler(handler)
    logger.setLevel(INFO)

df = pd.read_csv("http://daweb.ism.ac.jp/yosoku/materials/PF-example-data.txt", header=None)

df.columns = ["data"]

df.plot(figsize=(12,4))
plt.title("Test Data")

class ParticleFilter(object):
    def __init__(self, y, n_particle, sigma_2, alpha_2):
        self.y = y
        self.n_particle = n_particle
        self.sigma_2 = sigma_2
        self.alpha_2 = alpha_2
        self.log_likelihood = -np.inf

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def F_inv(self, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k+1

    """
    def F_inv2(self, k, i, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k[i] = np.max( idx[w_cumsum < u] ) + 1
        return k
    """

    def resampling2(self, weights):
        # re_start = time.time()
        """　計算量の少ない層化サンプリング　"""
        idx = np.asanyarray(range(self.n_particle))
        u0 = rd.uniform(0, 1/self.n_particle)
        u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        """ ↓ 約 0.1s * T だけの時間がかかる """
        k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
        """ ↓ 約 0.2s * T だけの時間がかかる
        k = np.asanyarray([0 for i in u])
        with ThreadPoolExecutor(max_workers=max_thread_num, thread_name_prefix="k_thread") as executor:
            count = 0
            for val in u:
                future = executor.submit(self.F_inv2, k, count, w_cumsum, idx, val)
                count += 1
        k = future.result()
        """
        # re_end = time.time()
        # print("resampling: %.3f seconds" % (re_end - re_start))
        return k

    def simulate(self, seed=71):
        rd.seed(seed)

        # 時系列データ数
        T = len(self.y)

        # 潜在変数
        x = np.zeros((T+1, self.n_particle))
        x_resampled = np.zeros((T+1, self.n_particle))

        # 潜在変数の初期値
        initial_x = rd.normal(0, 1, size=self.n_particle)
        x_resampled[0] = initial_x
        x[0] = initial_x

        # 重み
        w        = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T) # 時刻毎の尤度

        with ThreadPoolExecutor(max_workers=max_thread_num, thread_name_prefix="thread") as executor:
            # """
            for t in range(T):
                # cal_start = time.time()
                for i in range(max_thread_num):
                    future = executor.submit(self.calculation, i, t, x, x_resampled, w)
                    x, w = future.result()

                w_normed[t] = w[t]/np.sum(w[t]) # 規格化
                l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度

                # Resampling
                k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
                x_resampled[t+1] = x[t+1, k]
            # """

            """
            for t in range(T):
                cal_start = time.time()
                time_stamp = np.asanyarray([t for i in range(self.n_particle)])
                future = executor.map(self.calculation2, range(self.n_particle), time_stamp, x[t+1], x_resampled[t], w[t])
                result = np.asanyarray(list(future))

                # print("x[t]: ", x[t])
                # print("result_x: ", result[:, 0])
                x[t+1, :] = result[:, 0]
                w[t, :] = result[:, 1]

                w_normed[t] = w[t]/np.sum(w[t]) # 規格化
                l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度

                # Resampling
                k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
                x_resampled[t+1] = x[t+1, k]
                cal_end = time.time()
                print("calculation: %.3f seconds" % (cal_end - cal_start))
            """


        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - T*np.log(n_particle)

        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l


    def calculation(self, th_num, t, x, x_resampled, w):
        start = int(width * th_num)
        end = int(width * (th_num + 1) -1)
        # getLogger().info("%s start", t)
        for i in range(start, end):
          # 1階差分トレンドを適用
          v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
          x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
          w[t, i] = self.norm_likelihood(self.y[t], x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
        # getLogger().info("%s end", t)
        return x, w

    def calculation2(self, i, t, x, x_resampled, w):
        v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
        x = x_resampled + v # システムノイズの付加
        w = self.norm_likelihood(self.y[t], x, self.sigma_2) # y[t]に対する各粒子の尤度
        return x, w

    def get_filtered_value(self):
        """
        尤度の重みで加重平均した値でフィルタリングされた値を算出
        """
        return np.diag(np.dot(self.w_normed, self.x[1:].T))


"""### パーティクルフィルターによるフィルタリング"""

# ハイパーパラメーター
a = -2
b = -1

# n_particle = 10**3
n_particle = 10**3 * 5
sigma_2 = 2**a
alpha_2 = 10**b

max_thread_num = 1
width = int(n_particle / max_thread_num)

init_logger()
# getLogger().info("main start")
pf = ParticleFilter(df.data.values, n_particle, sigma_2, alpha_2)

start = time.time()
pf.simulate()
stop = time.time()
print('%.3f seconds' % (stop - start))
# getLogger().info("main end")

# pf.draw_graph()
