# %% import stuff
import trtoolbox.globalanalysis as ga
from test.data_generator import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# %% change ipython backend
try:
    get_ipython().run_line_magic('matplotlib', 'widget')
except NameError:
    pass

# %% generate data with parallel decaying processes
dgen = DataGenerator()
dgen.gen_data(
    wnlimit=[800, 1900],
    tcs=3,
    num_peaks=5,
    diff=True,
    avg_width=200,
    avg_std=15,
    noise=True,
    noise_scale=0.175,
    style='dec'
)

# %% plot generated data
dgen.print_tcs()
dgen.plot_sas()
dgen.plot_profile()
dgen.plot_data()

# %% do global fitting
sc = 5
rdnm = sc - sc * np.random.random(dgen.rate_constants.tcs.shape)
start_tcs = dgen.rate_constants.tcs * rdnm
res = ga.doglobalanalysis(dgen.data, dgen.time, dgen.wn, start_tcs, svds=3, style='dec')
res.plot_results()
res.plot_traces()
res.plot_spectra()
# plt.show()


# %% generate data with sequential decaying processes
plt.close('all')
dgen = DataGenerator()
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

# %% plot generated data
dgen.print_tcs()
dgen.plot_sas()
dgen.plot_profile()
dgen.plot_data()

# %% do global fitting
sc = 5
rdnm = sc - sc * np.random.random(dgen.rate_constants.tcs.shape)
start_tcs = dgen.rate_constants.tcs * rdnm
res = ga.doglobalanalysis(dgen.data, dgen.time, dgen.wn, start_tcs, svds=3, style='seq')
res.plot_results()
res.plot_traces()
res.plot_spectra()
# plt.show()


# %% generate data with parallel decaying processes with back reactions
plt.close('all')
dgen = DataGenerator()
dgen.gen_data(
    wnlimit=[800, 1900],
    tcs=3,
    num_peaks=5,
    diff=True,
    avg_width=200,
    avg_std=15,
    noise=True,
    noise_scale=0.175,
    style='back'
)

# %% plot generated data
dgen.print_tcs()
dgen.plot_sas()
dgen.plot_profile()
dgen.plot_data()

# %% do global fitting
sc = 5
rdnm = sc - sc * np.random.random(dgen.rate_constants.tcs.shape)
start_tcs = dgen.rate_constants.tcs * rdnm
res = ga.doglobalanalysis(dgen.data, dgen.time, dgen.wn, start_tcs, svds=3, style='back')
res.plot_results()
res.plot_traces()
res.plot_spectra()
plt.show()

# %% save files
# res.save_to_files(path)

# NOTE:
# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()

# %% Example for a branching of the photocycle due to heterogeneity. Here, we provide an own K-matrix and starting
# populations. Due to a different species, there are two parallel photoreactions which happen in parallel. In this
# example, there is one A -> B -> C -> GS and another D -> GS with A/D as same state but with different reaction
# pathways and GS as the ground state. Each reaction is described via a separate K-matrix but containing ALL of the
# states. These two matrices compromise a 3D matrix and the time constants should be shaped in the same way. The
# alpha values are the starting populations for each reaction.
#
# kmatrix1 = np.array([[-1, 0, 0, 0], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
# kmatrix2 = np.zeros((4, 4))
# kmatrix2[0, 0] = -1
# kmatrix2[1, 0] = 1
# kmatrix2[1, 1] = -1
# kmatrix = np.empty((4, 4, 2))
# kmatrix[:, :, 0] = kmatrix1
# kmatrix[:, :, 1] = kmatrix2
# del kmatrix1
# del kmatrix2
#
# alphas = np.array([0.8, 0.2])
#
# tc1 = np.array([5e-8, 1.8e-5, 3e-4, 2e0]).reshape(4,1)
# tc2 = np.array([5e-8, 0.9e-4, 0, 0]).reshape(4,1)
# tcs = np.hstack((tc1, tc2))
# del tc1
# del tc2
#
# ga_branched = mygf.doglobalanalysis(
#     data, time, wn,
#     tcs,
#     kmatrix=kmatrix, alphas=alphas,
#     svds=4, method='svd'
# )
