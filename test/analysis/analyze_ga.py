import cProfile
import pstats
import trtoolbox.globalanalysis as ga
from test.data_generator import DataGenerator
import numpy as np

dgen = DataGenerator()
profiler = cProfile.Profile()

total_time = 0
repititons = 50
for i in range(repititons):
    dgen.gen_data(
        wnlimit=[800, 1900],
        tcs=3,
        num_peaks=5,
        diff=True,
        avg_width=200,
        avg_std=15,
        noise=True,
        noise_scale=0.175,
        style='seq'
    )

    sc = 5
    rdnm = sc - sc * np.random.random(dgen.rate_constants.tcs.shape)
    start_tcs = dgen.rate_constants.tcs * rdnm

    profiler.enable()
    res = ga.doglobalanalysis(dgen.data, dgen.time, dgen.wn, start_tcs, svds=3, style='seq', silent=True)
    profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('tottime')
stats.print_stats('globalanalysis')
print('Average time of execution: ' + str(stats.total_tt/repititons))