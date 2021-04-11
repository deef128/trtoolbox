import cProfile
import pstats
import trtoolbox.lda as mylda
from test.data_generator import DataGenerator

dgen = DataGenerator()
profiler = cProfile.Profile()

total_time = 0
repititons = 20
for i in range(repititons):
    dgen.gen_data(
        wnlimit=[1500, 1800],
        num_peaks=4,
        diff=True,
        avg_width=100,
        avg_std=10,
        noise=True,
        noise_scale=0.15
    )

    profiler.enable()
    res = mylda.dolda(
        dgen.data,
        dgen.time,
        dgen.wn,
    )
    profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('tottime')
stats.print_stats('lda')
print('Average time of execution: ' + str(stats.total_tt/repititons))
