import cProfile
import pstats
import trtoolbox.expfit as expfit
import numpy as np
from test.data_generator import DataGenerator

dgen = DataGenerator()
profiler = cProfile.Profile()

total_time = 0
repititons = 100
for i in range(repititons):
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
    profiler.enable()
    res = expfit.dofit(dgen.data[50, :], dgen.time[0, :], init)
    profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumtime')
stats.print_stats('expfit')
print('Average time of execution: ' + str(stats.total_tt/repititons))
