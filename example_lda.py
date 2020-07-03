# %% import stuff
import trtoolbox.lda as mylda
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
    wnlimit=[1500, 1800],
    num_peaks=4,
    diff=True,
    avg_width=100,
    avg_std=10,
    noise=True,
    noise_scale=0.15
)

# %% plot generated data
dgen.print_tcs()
dgen.plot_das()
dgen.plot_profile()
dgen.plot_data()

# %% do lda
res = mylda.dolda(
    dgen.data,
    dgen.time,
    dgen.wn,
)
# res.plot_results()
res.plot_fitdata(alpha=0.9)
res.plot_ldamap(alpha=0.9)
res.plot_solutionvector()
res.plot_traces()
res.plot_spectra()
plt.show()

# %% save files
# res.save_to_files(path)

# NOTE:
# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
