import trtoolbox.mylda as mylda
import trtoolbox.mysvd as mysvd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

# data = mysvd.reconstruct(data, 10).svddata

res = mylda.dolda(data, time, wn, tlimits=[1e-6, 1e-1], tnum=50, seqmodel=True)
res.plot_results()
# res.plot_spectra(alpha=2)
# res.plot_traces(alpha=2)
res.plot_solutionvector(alpha=0.6)

# res2 = mylda.dolda(data, time, wn, seqmodel=True)
# res2.plot_results()

plt.show()

# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
