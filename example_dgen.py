from test.data_generator import DataGenerator
import matplotlib.pyplot as plt
from trtoolbox.plothelper import PlotHelper
import trtoolbox.mylda as mylda
import trtoolbox.myglobalfit as mygf

phelper = PlotHelper()
dgen = DataGenerator()
dgen.gen_data(noise=False)
for i in range(len(dgen.tcs)):
    print('%i. forward: %e, backward: %e' % (i, dgen.tcs[i, 0], dgen.tcs[i, 1]))

# plt.figure()
# plt.plot(dgen.wn, dgen.das)

plt.figure()
plt.plot(dgen.time.T, dgen.profile)
plt.xscale('log')

# phelper.plot_heatmap(dgen.data, dgen.time, dgen.wn)

res = mygf.doglobalfit(dgen.data, dgen.time, dgen.wn, [[4e-6, 3e-4, 2e-2], [1e1, 1e-4, 1e-2]], method='raw', back=True)
res.plot_profile()
res = mygf.doglobalfit(dgen.data, dgen.time, dgen.wn, [[4e-6, 3e-4, 2e-2], [1e1, 1e-4, 1e-2]], method='svd', back=True)
res.plot_profile()

# res = mylda.dolda(dgen.data, dgen.time, dgen.wn, alimits=[0.1, 50])
# res.plot_results()
# # res.plot_traces()
# # res.plot_spectra()
# res.plot_solutionvector()

plt.show()
