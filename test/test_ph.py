import unittest
import numpy as np
import matplotlib.pyplot as plt
from trtoolbox.plothelper import PlotHelper
from data_generator import DataGenerator


class TestGF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # generate data
        dgen = DataGenerator()
        dgen.gen_data()
        cls.data = dgen.data
        cls.wn = dgen.wn
        cls.time = dgen.time
        cls.tcs = dgen.rate_constants.tcs
        cls.ks = dgen.rate_constants.ks
        cls.rate_constants = dgen.rate_constants
        cls.profile = dgen.profile
        cls.das = dgen.sas

        dgen.gen_data(style='back')
        cls.tcs_back = dgen.rate_constants.tcs
        cls.rate_constants_back = dgen.rate_constants

        cls.ph = PlotHelper()

        # prevents plt.show() from blocking execution
        plt.ion()

    def test_heatmap_interpolate(self):
        self.ph.plot_heatmap(self.data, self.time, self.wn, interpolate=True)

    def test_surface_interpolate(self):
        self.ph.plot_surface(self.data, self.time, self.wn, interpolate=True, step=5)