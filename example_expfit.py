import trtoolbox.expfit as expfit
import matplotlib.pyplot as plt
import numpy as np
from test.data_generator import DataGenerator

dgen = DataGenerator()
dgen.gen_data(
    wnlimit=[400, 700],
    tlimit=[1e-8, 1e0],
    tcs=[1e-7, 1e-5, 1e-3, 1e-1],
    num_peaks=np.array([[530, 620, 450, 550]]),
    avg_width=100,
    avg_std=15,
    noise=True
)

init = np.array([[-1, 1e-6], [1, 1e-2]])
res = expfit.dofit(dgen.data[50, :], dgen.time[0, :], init)

res.plot_result_traces()
res.plot_result()
plt.show()