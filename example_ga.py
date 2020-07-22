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

# %% generate data
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
dgen.plot_das()
dgen.plot_profile()
dgen.plot_data()

# %% do global fitting
sc = 5
rdnm = sc - sc * np.random.random(dgen.rate_constants.tcs.shape)
start_tcs = dgen.rate_constants.tcs * rdnm
res = ga.doglobalanalysis(dgen.data, dgen.time, dgen.wn, start_tcs, svds=3, style='seq', method='svd')
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
