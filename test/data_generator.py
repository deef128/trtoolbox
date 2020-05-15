# TODO: finish
import numpy as np
import numpy.random as nrand
from scipy import signal
from trtoolbox.myglobalfit import create_profile


class DataGenerator:

    def __init__(self):
        self.time = np.array([])
        self.wn = np.array([])
        self.data = np.array([])
        self.das = np.array([])
        self.tcs = np.array([])
        self.profile = np.array([])

    def gen_time(self):
        self.time = np.logspace(-7, -1, 500)
        self.time = self.time.reshape((1, self.time.size))

    def gen_wn(self):
        self.wn = np.linspace(1500, 1700, num=1700-1500+1)
        self.wn = self.wn.reshape((self.wn.size, 1))

    def gen_das(self):
        num_das = 3
        num_peaks = 1
        avg_width = 30
        avg_std = 5
        self.das = np.zeros((self.wn.shape[0], num_das))

        for i in range(num_das):
            gaus = signal.gaussian(avg_width, std=avg_std, sym=False)
            pos = nrand.randint(0, high=self.wn.shape[0]-avg_width)
            das = np.zeros(self.wn.shape[0])
            das[pos:pos+avg_width] = gaus
            self.das[:, i] = das.T

    def gen_data_das(self):
        num_tcs = 3
        expo = -1 * nrand.choice(np.arange(2, 6, step=1), size=(num_tcs,), replace=False)
        expo = np.sort(expo)
        pre = -9 * nrand.random(size=(num_tcs,)) + 9
        # self.tcs = np.array([pre[i]*10.**expo[i] for i in range(num_tcs)])
        self.tcs = np.array([[1e-6, 1e-4, 1e-2],
                            [1e1, 1e-4, 1e1]]
                            ).T
        self.profile = create_profile(self.time, 1/self.tcs, back=True)
        self.data = self.das.dot(self.profile.T)

    def gen_data(self, noise=False, noise_scale=0.1):
        self.gen_time()
        self.gen_wn()
        self.gen_das()
        self.gen_data_das()
        if noise is True:
            self.data = self.data + nrand.normal(0, scale=noise_scale, size=self.data.shape)
