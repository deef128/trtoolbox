import unittest
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.lda as mylda
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
        cls.profile = dgen.profile
        cls.das = dgen.sas

        # prevents plt.show() from blocking execution
        plt.ion()

    def test_check_input(self):
        data, time, wn = mylda.check_input(self.data, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)

        data, time, wn = mylda.check_input(self.data, self.time.T, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(wn.shape, self.wn.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])

        data, time, wn = mylda.check_input(self.data, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mylda.check_input(self.data.T, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape, self.time.shape)
        self.assertEqual(wn.shape, self.wn.shape)

        data, time, wn = mylda.check_input(self.data.T, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mylda.check_input(
            self.data,
            self.time.flatten(),
            self.wn.flatten()
        )
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        with self.assertRaises(ValueError):
            data, time, wn = mylda.check_input(
                self.data,
                self.time,
                self.wn[2:]
            )

        with self.assertRaises(ValueError):
            data, time, wn = mylda.check_input(
                self.data,
                self.time[2:],
                self.wn
            )

        with self.assertRaises(ValueError):
            data, time, wn = mylda.check_input(
                self.data[2:, :-3],
                self.time,
                self.wn
            )

    def test_gen_taus(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 50)
        self.assertEqual(taus[0], 1e-6)

        taus = mylda.gen_taus(1e-4, 1e1, 70)
        self.assertEqual(taus[-1], 1e1)
        self.assertEqual(len(taus), 70)

        taus = mylda.gen_taus(1e-1, 1e-7, 150)
        self.assertEqual(taus[0], 1e-7)
        self.assertEqual(taus[-1], 1e-1)

        with self.assertRaises(ValueError):
            taus = mylda.gen_taus(1e-4, 1e-4, 3)

        taus = mylda.gen_taus(1e-5, 1e-2, -43)
        self.assertEqual(len(taus), 100)

    def test_gen_dmatrix(self):
        taus = mylda.gen_taus(1e-5, 1e1, 100)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        self.assertEqual(dmatrix.shape, (self.time.size, taus.size))

    def test_gen_lamtrix(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 200)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        lmatrix = mylda.gen_lmatrix(dmatrix)
        self.assertEqual(lmatrix.shape, (dmatrix.shape[1], dmatrix.shape[1]))

    def test_gen_alphas(self):
        alphas = mylda.gen_alphas(1e-1, 1e1, 200)
        self.assertEqual(alphas[0], 1e-1)
        self.assertEqual(alphas[-1], 1e1)
        self.assertEqual(len(alphas), 200)

        alphas = mylda.gen_alphas(1e1, 1e-1, 200)
        self.assertEqual(alphas[0], 1e-1)
        self.assertEqual(alphas[-1], 1e1)

        with self.assertRaises(ValueError):
            alphas = mylda.gen_alphas(1, 1, 113)

    def test_inversesvd(self):
        matrix = np.random.rand(100, 500)
        inverse = mylda.inversesvd(matrix)
        np.testing.assert_array_almost_equal(
            matrix.dot(inverse),
            np.identity(100)
        )

    def test_tik(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 200)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        x_k = mylda.tik(self.data, dmatrix, 0.1)
        fitdata = np.transpose(dmatrix.dot(x_k))
        np.testing.assert_array_almost_equal(
            self.data,
            fitdata,
            1
        )

    def test_tiks(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 200)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        alphas = mylda.gen_alphas(1e-1, 1e1, 100)
        x_ks = mylda.tiks(self.data, dmatrix, alphas)
        self.assertEqual(x_ks.shape[2], len(alphas))

    def test_calc_lcurve(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 100)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        lmatrix = mylda.gen_lmatrix(dmatrix)
        alphas = mylda.gen_alphas(1e-1, 1e1, 50)
        x_ks = mylda.tiks(self.data, dmatrix, alphas)
        lcurve = mylda.calc_lcurve(self.data, dmatrix, lmatrix, x_ks)
        self.assertEqual(lcurve.shape[0], len(alphas))

    def test_tik_lstsq(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 200)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        x_k_lstsq = mylda.tik_lstsq(self.data, dmatrix, 1)
        x_k = mylda.tik(self.data, dmatrix, 1)
        np.testing.assert_array_almost_equal(
            x_k_lstsq,
            x_k,
        )

    def test_tsvd(self):
        taus = mylda.gen_taus(1e-6, 1e-1, 200)
        dmatrix = mylda.gen_dmatrix(self.time, taus)
        x_k_tsvd = mylda.tsvd(self.data, dmatrix, 10)
        x_k_tik = mylda.tik(self.data, dmatrix, 1)
        np.testing.assert_array_almost_equal(
            x_k_tsvd,
            x_k_tik,
            0
        )

    def test_dolda(self):
        res_tik = mylda.dolda(self.data, self.time, self.wn)
        res_tsvd = mylda.dolda(self.data, self.time, self.wn, k=10)

        np.testing.assert_array_almost_equal(
            res_tik.fitdata,
            res_tsvd.fitdata
        )
