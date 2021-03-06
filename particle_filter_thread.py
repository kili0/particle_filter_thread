import numpy as np
import numpy.random as rd
import pandas as pd

import time
from concurrent.futures import ThreadPoolExecutor
from logging import StreamHandler, Formatter, INFO, getLogger

df = pd.read_csv("http://daweb.ism.ac.jp/yosoku/materials/PF-example-data.txt", header=None)
df.columns = ["data"]

def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger = getLogger()
    logger.addHandler(handler)
    logger.setLevel(INFO)

class ParticleFilter(object):
    def __init__(self, y, n_particle, sigma_2, alpha_2):
        self.y = y
        self.n_particle = n_particle
        self.sigma_2 = sigma_2
        self.alpha_2 = alpha_2
        self.log_likelihood = -np.inf

        self.time_count = time_count
        self.resampling_time = resampling_time
        self.cal_time = cal_time

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def F_inv(self, w_cumsum, idx, u):
        if np.any(w_cumsum < u) == False:
            return 0
        k = np.max(idx[w_cumsum < u])
        return k+1

    def resampling2(self, weights):
        re_start = time.time()
        """　計算量の少ない層化サンプリング　"""
        idx = np.asanyarray(range(self.n_particle))
        u0 = rd.uniform(0, 1/self.n_particle)
        u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(weights)
        k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
        re_end = time.time()
        self.resampling_time[self.time_count] = re_end - re_start
        self.time_count += 1
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

        """
        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

        with ThreadPoolExecutor(max_workers=max_thread_num, thread_name_prefix="thread") as executor:
            for t in range(T):
                cal_start = time.time()
                for i in range(max_thread_num):
                    future = executor.submit(self.calculate, t, i)

                flag = future.result()
                w_normed[t] = w[t]/np.sum(w[t]) # 規格化
                l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度
                cal_end = time.time()
                self.cal_time[self.time_count] = cal_end - cal_start

                # Resampling
                k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
                x_resampled[t+1] = x[t+1, k]
        """
        for t in range(T):
            cal_start = time.time()
            for i in range(self.n_particle):
                v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
                x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
                w[t, i] = self.norm_likelihood(self.y[t], x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
            w_normed[t] = w[t]/np.sum(w[t]) # 規格化
            l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度
            cal_end = time.time()
            self.cal_time[self.time_count] = cal_end - cal_start

            # Resampling
            k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
            x_resampled[t+1] = x[t+1, k]

        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - T*np.log(n_particle)

        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

    def calculate(self, t, th_num):
        start = int(width * th_num)
        end = int(width * (th_num + 1) -1)
        for i in range(start, end):
            # 1階差分トレンドを適用
            v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
            self.x[t+1, i] = self.x_resampled[t, i] + v # システムノイズの付加
            self.w[t, i] = self.norm_likelihood(self.y[t], self.x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
        return 1

    def get_filtered_value(self):
        """ 尤度の重みで加重平均した値でフィルタリングされた値を算出 """
        return np.diag(np.dot(self.w_normed, self.x[1:].T))


""" パーティクルフィルターによるフィルタリング """

# ハイパーパラメーター
a = -2
b = -1

n_particle = 10**3 * 5
# n_particle = 10**3 * 5
sigma_2 = 2**a
alpha_2 = 10**b

max_thread_num = 4
width = int(n_particle / max_thread_num)

""" for debug """
time_len = 100
resampling_time = np.zeros(time_len)
cal_time = np.zeros(time_len)
time_count = 0

init_logger()
# getLogger().info("main start")
pf = ParticleFilter(df.data.values, n_particle, sigma_2, alpha_2)

start = time.time()
pf.simulate()
stop = time.time()
print('%.3f seconds' % (stop - start))

# print("resampling time: max = %.3f seconds" % np.max(resampling_time))
# print("resampling time: min = %.3f seconds" % np.min(resampling_time))
print("resampling time: mid = %.3f seconds" % (np.sum(resampling_time)/time_len))
print("calculation time: mid = %.3f seconds" % (np.sum(cal_time)/time_len))

# getLogger().info("main end")
