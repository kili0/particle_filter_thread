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

        self.T = T
        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l

        self.time_count = time_count
        self.resampling_time = resampling_time
        self.cal_time = cal_time

    def norm_likelihood(self, y, x, s2):
        return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

    def task(self, i, t):
        # getLogger().info("submit")
        # print("x: ", self.x)
        # 1階差分トレンドを適用
        v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
        self.x[t+1, i] = self.x_resampled[t, i] + v # システムノイズの付加
        self.w[t, i] = self.norm_likelihood(self.y[t], self.x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
        return 1

    def main(self, seed=71):
        rd.seed(seed)
        start = time.time()
        init_logger()

        with ThreadPoolExecutor(max_workers=max_thread_num, thread_name_prefix="thread") as executor:
            for t in range(T):
                cal_start = time.time()
                for i in range(self.n_particle):
                    future = executor.submit(self.task, i, t)

                flag = future.result()
                w_normed[t] = w[t]/np.sum(w[t]) # 規格化
                l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度
                cal_end = time.time()
                self.cal_time[self.time_count] = cal_end - cal_start
                self.time_count += 1


if __name__ == "__main__":
    # ハイパーパラメーター
    a = -2
    b = -1

    n_particle = 4 * 10**5
    # n_particle = 10**3 * 5
    sigma_2 = 2**a
    alpha_2 = 10**b

    # T = len(df.data.values)
    T = 1
    x = np.zeros((T+1, n_particle))
    x_resampled = np.zeros((T+1, n_particle))
    initial_x = rd.normal(0, 1, size=n_particle)
    x_resampled[0] = initial_x
    x[0] = initial_x
    w        = np.zeros((T, n_particle))
    w_normed = np.zeros((T, n_particle))
    l = np.zeros(T) # 時刻毎の尤度

    max_thread_num = 1
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
    pf.main()
    stop = time.time()
    # print('%.3f seconds' % (stop - start))
    print("calculation time mid: %.4f seconds" % (np.sum(cal_time)/T))
