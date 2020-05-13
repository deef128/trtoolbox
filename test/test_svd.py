import unittest
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd


class TestSVD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt('./data/data.dat', delimiter=',')
        cls.wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
        cls.time = np.loadtxt('./data/time.dat', delimiter=',')
        # prevents plt.show() from blocking execution
        plt.ion()

    def test_check_input(self):
        data, time, wn = mysvd.check_input(self.data, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)

        data, time, wn = mysvd.check_input(self.data, self.time.T, self.wn)
        self.assertEqual(time.shape[1], self.data.shape[1])

        data, time, wn = mysvd.check_input(self.data, self.time.T, self.wn.T)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

        data, time, wn = mysvd.check_input(self.data.T, self.time, self.wn)
        self.assertEqual(data.shape, self.data.shape)

        data, time, wn = mysvd.check_input(self.data.T, self.time.T, self.wn.T)
        self.assertEqual(data.shape, self.data.shape)
        self.assertEqual(time.shape[1], self.data.shape[1])
        self.assertEqual(wn.shape[0], self.data.shape[0])

    def test_wrapper_svd(self):
        u, s, vt = mysvd.wrapper_svd(self.data)
        # checking dimensions
        self.assertEqual(u.shape[0], self.data.shape[0])
        self.assertEqual(s.shape[0], self.data.shape[0])
        self.assertEqual(vt.shape[0], self.data.shape[1])

        tdata = np.transpose(self.data)
        u, s, vt = mysvd.wrapper_svd(tdata)
        # checking dimensions
        self.assertEqual(u.shape[0], tdata.shape[0])
        self.assertEqual(s.shape[0], self.data.shape[0])
        self.assertEqual(vt.shape[0], tdata.shape[1])

    def test_reconstruct(self):
        res = mysvd.reconstruct(self.data, 5)
        # checking dimensions
        self.assertEqual(res.u.shape[0], self.data.shape[0])
        self.assertEqual(res.s.shape[0], self.data.shape[0])
        self.assertEqual(res.vt.shape[0], self.data.shape[1])
        self.assertEqual(res.svddata.shape, self.data.shape)

        tdata = np.transpose(self.data)
        res = mysvd.reconstruct(tdata, 5)
        # checking dimensions
        self.assertEqual(res.u.shape[0], tdata.shape[0])
        self.assertEqual(res.s.shape[0], self.data.shape[0])
        self.assertEqual(res.vt.shape[0], tdata.shape[1])
        self.assertEqual(res.svddata.shape, tdata.shape)

    def test_show_svs(self):
        mysvd.show_svs(self.data, self.time, self.wn)
        plt.close('all')
        mysvd.show_svs(np.transpose(self.data), self.time, self.wn)
        plt.close('all')

        mysvd.show_svs(self.data.T, self.time, self.wn)
        plt.close('all')
        mysvd.show_svs(self.data, self.time.T, self.wn)
        plt.close('all')

    def test_dosvd(self):
        mysvd.dosvd(self.data, self.time, self.wn, 5)
        plt.close('all')
        n = [1, 2, 3, 5]
        mysvd.dosvd(self.data, self.time, self.wn, n)
        plt.close('all')
        n = np.array(n)
        mysvd.dosvd(self.data, self.time, self.wn, n)
        plt.close('all')

        tdata = np.transpose(self.data)
        mysvd.dosvd(tdata, self.time, self.wn, 5)
        plt.close('all')
        n = [1, 2, 3, 5]
        mysvd.dosvd(tdata, self.time, self.wn.T, n)
        plt.close('all')
        n = np.array(n)
        mysvd.dosvd(tdata, self.time.T, self.wn, n)
        plt.close('all')     
