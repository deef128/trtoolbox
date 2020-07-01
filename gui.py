import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd
import trtoolbox.myglobalfit as mygf
import trtoolbox.mylda as mylda
from test.data_generator import DataGenerator
from trtoolbox.plothelper import PlotHelper
from PyQt5 import QtWidgets, uic

Ui_MainWindow, QtBaseClass = uic.loadUiType("gui/interface.ui")

Ui_Dialog, _ = uic.loadUiType('gui/dialog_dgen.ui')


class DataStorage():
    """ Object for data storage.

    Attributes
    ----------
    type : str
        Object type.
    _data : np.array
        Data matrix.
    _time : np.array
        Time array.
    _wn : np.array.
        Frequency array.
    _phelper : PlotHelper()
        Contains plotting routines.
    wn_name : str
        Name of the frequency unit (default: wavenumber).
    wn_unit : str
        Frequency unit (default cm^{-1}).
    t_name : str
        Time name (default: time).
    time_unit : str
        Time uni (default: s).
    trunc_time : list
        Limits for truncation in time units
    trunc.wn : list
        Limit for truncation in freq units.
    """

    def __init__(self):
        self.type = 'raw'
        self._data = np.array([])
        self._time = np.array([])
        self._wn = np.array([])
        self._phelper = PlotHelper()
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.time_name = 'time'
        self.time_unit = 's'
        self.trunc_time = list()
        self.trunc_wn = list()

    @property
    def wn(self):
        """ wn getter.

        Returns
        -------
        self._wn : np.array
            Truncated freq array.
        """

        i1 = np.argmin(abs(self._wn - self.trunc_wn[0]))
        i2 = np.argmin(abs(self._wn - self.trunc_wn[1]))
        return self._wn[i1:i2+1, 0:1]

    @wn.setter
    def wn(self, val):
        """ wn setter. It also sets trunc_wn to the min/max values.
        """

        if val.dtype != 'float':
            val = val.astype('float64')
        self._wn = val.reshape((val.size, 1))
        self.trunc_wn = [np.min(self._wn), np.max(self._wn)]

    @property
    def time(self):
        """ time getter.

        Returns
        -------
        self._time : np.array
            Truncated time array.
        """

        i1 = np.argmin(abs(self._time - self.trunc_time[0]))
        i2 = np.argmin(abs(self._time - self.trunc_time[1]))
        return self._time[0:1, i1:i2+1]

    @time.setter
    def time(self, val):
        """ time setter. It also sets trunc_time to the min/max values.
        """

        if val.dtype != 'float':
            val = val.astype('float64')
        self._time = val.reshape((1, val.size))
        self.trunc_time = [np.min(self._time), np.max(self._time)]

    @property
    def data(self):
        """ data getter.

        Returns
        -------
        self._data : np.array
            Truncated data matrix.
        """

        w1 = np.argmin(abs(self._wn - self.trunc_wn[0]))
        w2 = np.argmin(abs(self._wn - self.trunc_wn[1]))
        t1 = np.argmin(abs(self._time - self.trunc_time[0]))
        t2 = np.argmin(abs(self._time - self.trunc_time[1]))
        return self._data[w1:w2+1, t1:t2+1]

    @data.setter
    def data(self, val):
        """ data setter.
        """

        if val.dtype != 'float':
            val = val.astype('float64')
        self._data = val

    def check(self):
        """Checks if data was loaded

        Returns
        -------
        bool : bool
            True if data was loaded
        """

        if self.data.size == 0 or self.time.size == 0 or self.wn.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Please load data.')
            return False
        return True

    def init_phelper(self):
        """ Initiliazes phelper after clean().
        """

        if type(self._phelper) == list:
            self._phelper = PlotHelper()

    def plot_data(self):
        """ Plots a contour map.
        """

        self.init_phelper()
        self._phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Raw Data', newfig=True)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_data_3d(self):
        """ Plots a 3D map.
        """

        self.init_phelper()
        self._phelper.plot_surface(
            self.data, self.time, self.wn,
            title='Raw Data')
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_spectra(self):
        """ Plots interactive spectra.
        """

        self.init_phelper()
        self._phelper.plot_spectra(self)

    def plot_traces(self):
        """ Plots interactive time traces.
        """

        self.init_phelper()
        self._phelper.plot_traces(self)


class Results():
    """ Container for results objects.

    Attributes
    ----------
    _phelper : PlotHelper()
        Contains plotting routines.
    svd : mysvd.Results()
        SVD results.
    gf : mygf.Results()
        Global Fit results.
    lda : mylda.Results()
        LDA results.
    """

    def __init__(self):
        self._phelper = PlotHelper()
        self.svd = mysvd.Results()
        self.gf = mygf.Results()
        self.lda = mylda.Results()

    def check_svd(self, silent=False):
        """Checks if SVD was performed

        Parameters
        ----------
        silent : bool, optional
            Error message supressed if True, by default False

        Returns
        -------
        bool : bool
            True if SVD was performed
        """

        if self.svd.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do SVD.')
            return False
        return True

    def check_gf(self, silent=False):
        """Checks if global fitting was performed

        Parameters
        ----------
        silent : bool, optional
            Error message supressed if True, by default False

        Returns
        -------
        bool : bool
            True if global fitting was performed
        """

        if self.gf.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do global fit analysis.')
            return False
        return True

    def check_lda(self, silent=False):
        """Checks if LDA was performed

        Parameters
        ----------
        silent : bool, optional
            Error message supressed if True, by default False

        Returns
        -------
        bool : bool
            True if LDA was performed
        """

        if self.lda.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do LDA.')
            return False
        return True


class DgenDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super(DgenDialog, self).__init__()
        self.setupUi(self)

    def get_results(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            # get all values
            txt_time = self.txt_time.text()
            txt_wn = self.txt_wn.text()
            sb_taus = self.sb_taus.value()
            sb_peaks = self.sb_peaks.value()
            txt_width = self.txt_width.text()
            txt_std = self.txt_std.text()
            cb_diff = self.cb_diff.isChecked()
            cb_back = self.cb_back.isChecked()
            cb_noise = self.cb_noise.isChecked()
            txt_noise = self.txt_noise.text()

            dgen = DataGenerator()
            dgen.gen_data(
                tlimit=[float(s) for s in txt_time.split(',')],
                wnlimit=[int(s) for s in txt_wn.split(',')],
                tcs=[-1 for i in range(sb_taus)],
                num_peaks=sb_peaks,
                avg_width=float(txt_width),
                avg_std=float(txt_std),
                diff=cb_diff,
                back=cb_back,
                noise=cb_noise,
                noise_scale=float(txt_noise)
            )
            return dgen
        else:
            return None


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.data = DataStorage()
        self.results = Results()

        self.actionLoad_all.triggered.connect(self.get_file_all)
        self.actionLoad_time.triggered.connect(self.get_file_time)
        self.actionLoad_frequency.triggered.connect(self.get_file_wn)
        self.actionLoad_data.triggered.connect(self.get_file_data)
        self.actionLoad_test_data.triggered.connect(self.get_test_data)

        self.actionTime_change.triggered.connect(self.change_time)
        self.actionFreq_change.triggered.connect(self.change_freq)

        self.pb_plt_close.clicked.connect(self.close_all)
        self.pb_save_results.clicked.connect(self.save_all)

        self.pb_truncate.clicked.connect(self.truncate)
        self.pb_data_plot.clicked.connect(self.plot_raw)

        self.pb_dosvd.clicked.connect(self.dosvd)
        self.pb_singular.clicked.connect(self.show_svs)
        self.pb_svd_plot.clicked.connect(self.plot_svd)

        self.pb_dogf.clicked.connect(self.dogf)
        self.pb_gf_plot.clicked.connect(self.plot_gf)

        self.pb_lda_dolda.clicked.connect(self.dolda)
        self.pb_lda_plot.clicked.connect(self.plot_lda)

    def get_file_all(self):
        """ Loads all files.
        """

        self.get_file_time()
        self.get_file_wn()
        self.get_file_data()

    def get_file_time(self):
        """ Loads time file. Delimiter should be ','.
             It also sets text into the truncation field.
        """

        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open time file')
        if os.path.exists(fname[0]):
            try:
                self.data.time = np.loadtxt(fname[0], delimiter=',')
                self.label_file_time.setText(
                    'Time file:\n' + fname[0].split(os.path.sep)[-1][-22:])
                str_trunc = '%.1e, %.1e' % (tuple(self.data.trunc_time))
                self.txt_trunc_time.setText(str_trunc)
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_file_wn(self):
        """ Loads frequency file. Delimiter should be ','.
            It also sets text into the truncation field.
        """

        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open frequency file'
        )
        if os.path.exists(fname[0]):
            try:
                self.data.wn = np.loadtxt(fname[0], delimiter=',')
                self.label_file_freq.setText(
                    'Freq file:\n' + fname[0].split(os.path.sep)[-1][-22:])
                str_trunc = '%.1f, %.1f' % (tuple(self.data.trunc_wn))
                self.txt_trunc_freq.setText(str_trunc)
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_file_data(self):
        """ Loads data file. Delimiter should be ','.
        """

        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open data file')
        if os.path.exists(fname[0]):
            try:
                self.data.data = np.loadtxt(fname[0], delimiter=',')
                self.label_file_data.setText(
                    'Data file:\n...' + fname[0][-50:]
                )
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_test_data(self):
        """Generates test data.
        """

        d = DgenDialog()
        dgen = d.get_results()
        if dgen:
            self.data.__init__()

            self.data.time = dgen.time
            str_trunc = '%.1e, %.1e' % (tuple(self.data.trunc_time))
            self.txt_trunc_time.setText(str_trunc)
            self.label_file_time.setText(
                'Time file:\nGenerated time'
            )

            self.data.wn = dgen.wn
            str_trunc = '%.1f, %.1f' % (tuple(self.data.trunc_wn))
            self.txt_trunc_freq.setText(str_trunc)
            self.label_file_freq.setText(
                'Freq file:\nGenerated wavenumbers'
            )

            self.data.data = dgen.data
            str_taus = ''
            if dgen.back is False:
                for t in dgen.tcs:
                    str_taus = str_taus + format(t, '.1e') + ', '
            elif dgen.back is True:
                for row in dgen.tcs.transpose():
                    for t in row:
                        str_taus = str_taus + format(t, '.1e') + ', '
                    str_taus = str_taus[:-2] + ' / '
            self.label_file_data.setText(
                'Generated data with taus:\n' + str_taus[:-2]
            )

            plt.close('all')
            self.data.plot_data()
            dgen.plot_profile()
            plt.show()

    def change_time(self):
        """ Open dialogs to change time name und unit.
        """

        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter time unit long:'
        )
        if ok:
            self.data.time_name = text
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter time unit short:'
        )
        if ok:
            self.data.time_unit = text

    def change_freq(self):
        """ Open dialogs to change frequency name und unit.
        """

        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter frequency unit long:'
        )
        if ok:
            self.data.wn_name = text
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter frequency unit short:'
        )
        if ok:
            self.data.wn_unit = text

    def close_all(self):
        """ Closes all plots.
        """

        plt.close('all')

    def save_all(self):
        """ Saves results to ASCII files.
        """

        basepath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory"
            )
        )
        if self.results.check_svd(silent=True) is True:
            path = os.path.join(basepath, 'svd')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.svd.save_to_files(path)
        if self.results.check_gf(silent=True) is True:
            path = os.path.join(basepath, 'gf')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.gf.save_to_files(path)
        if self.results.check_lda(silent=True) is True:
            try:
                alpha = float(self.txt_lda_plot_alpha.text())
            except ValueError:
                print('Please enter a valid float')
                return
            path = os.path.join(basepath, 'lda')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.lda.save_to_files(path, alpha=alpha)

    def truncate(self):
        """ Updates truncation values.
        """

        trunc_time = self.txt_trunc_time.text()
        self.data.trunc_time = [float(s) for s in trunc_time.split(',')]
        trunc_wn = self.txt_trunc_freq.text()
        self.data.trunc_wn = [float(s) for s in trunc_wn.split(',')]

    def plot_raw(self):
        """ Plots raw data.
        """

        if self.data.check() is False:
            return

        if self.cb_data_contour.isChecked() is True:
            self.data.plot_data()
        if self.cb_data_3d.isChecked() is True:
            self.data.plot_data_3d()
        if self.cb_data_spectra.isChecked() is True:
            self.data.plot_spectra()
        if self.cb_data_traces.isChecked() is True:
            self.data.plot_traces()
        plt.show()

    def show_svs(self):
        """ Shows singular components.
        """

        if self.data.check() is False:
            return

        mysvd.show_svs(self.data.data, self.data.time, self.data.wn)
        plt.show()

    def dosvd(self):
        """ Performes SVD.
        """

        if self.data.check() is False:
            return

        num = self.sb_svd_svd.value()
        self.results.svd = mysvd.dosvd(
            self.data.data,
            self.data.time,
            self.data.wn,
            n=num
        )
        self.results.svd.plot_results()
        plt.show()

    def plot_svd(self):
        """ Plots SVD results.
        """

        if self.results.check_svd() is False:
            return

        if self.cb_svd_contour.isChecked() is True:
            self.results.svd.plot_svddata(newfig=True)
        if self.cb_svd_3d.isChecked() is True:
            self.results.svd.plot_svddata_3d()
        if self.cb_svd_spectra.isChecked() is True:
            self.results.svd.plot_spectra()
        if self.cb_svd_traces.isChecked() is True:
            self.results.svd.plot_traces()
        plt.show()

    def dogf(self):
        """ Performs global fitting.
        """

        if self.data.check() is False:
            return

        num = self.sb_gf_svd.value()
        tcs_str = self.txt_gf_tcs.toPlainText()
        tcs_str = tcs_str.strip(' ')
        tcs_str = tcs_str.rstrip(',')
        try:
            tcs = [float(i) for i in tcs_str.split(",")]
        except ValueError:
            print('Please provide valid string')
            return
        self.results.gf = mygf.doglobalfit(
            self.data.data,
            self.data.time,
            self.data.wn,
            tcs,
            svds=num
        )
        self.results.gf.plot_results()
        plt.show()
        self.print_results()

    def print_results(self):
        """ Prints results to text field.
        """

        str = ''
        for i, t in enumerate(self.results.gf.tcs):
            str = str + '%i. %.2e (%.2e)\n' % (i+1, t, self.results.gf.var[i])
        str = str + 'R^2 = %.2f%%' % (self.results.gf.r2)
        self.txt_gf_results.clear()
        self.txt_gf_results.insertPlainText(str)

    def plot_gf(self):
        """ Plots global fit results.
        """

        if self.results.check_gf() is False:
            return

        if self.cb_gf_contour.isChecked() is True:
            self.results.gf.plot_fitdata()
        if self.cb_gf_3d.isChecked() is True:
            self.results.gf.plot_fitdata_3d()
        if self.cb_gf_spectra.isChecked() is True:
            self.results.gf.plot_spectra()
        if self.cb_gf_traces.isChecked() is True:
            self.results.gf.plot_traces()
        if self.cb_gf_profile.isChecked() is True:
            self.results.gf.plot_profile()
        if self.cb_gf_das.isChecked() is True:
            self.results.gf.plot_das()
        plt.show()

    def dolda(self):
        """ Performs LDA.
        """

        if self.data.check() is False:
            return

        tcs = []
        alpha = []

        try:
            tcs.append(float(self.txt_lda_tc_bottom.text()))
            tcs.append(float(self.txt_lda_tc_upper.text()))
            tcs_num = int(self.txt_lda_tc_number.text())

            alpha.append(float(self.txt_lda_alpha_bottom.text()))
            alpha.append(float(self.txt_lda_alpha_upper.text()))
            alpha_num = int(self.txt_lda_alpha_number.text())
        except ValueError:
            print('Just type numbers.')

        if self.cb_lda_use_svd.isChecked() is True:
            if self.results.check_svd() is True:
                data = self.results.svd.svddata
            else:
                return
        else:
            data = self.data.data

        self.results.lda = mylda.dolda(
            data,
            self.data.time,
            self.data.wn,
            tlimits=tcs,
            tnum=tcs_num,
            alimits=alpha,
            anum=alpha_num
        )

        self.results.lda.plot_results()
        plt.show()

    def plot_lda(self):
        """ Plots LDA results.
        """

        if self.results.check_lda() is False:
            return

        if self.cb_lda_results.isChecked() is True:
            self.results.lda.plot_results()

        try:
            alpha = float(self.txt_lda_plot_alpha.text())
        except ValueError:
            print('Please enter a valied float')
            return

        if self.cb_lda_contour.isChecked() is True:
            self.results.lda.plot_fitdata(alpha=alpha)
        if self.cb_lda_3d.isChecked() is True:
            self.results.lda.plot_fitdata_3d()
        if self.cb_lda_spectra.isChecked() is True:
            self.results.lda.plot_spectra(alpha=alpha)
        if self.cb_lda_traces.isChecked() is True:
            self.results.lda.plot_traces(alpha=alpha)
        if self.cb_lda_map.isChecked() is True:
            self.results.lda.plot_ldamap(alpha=alpha)
        if self.cb_lda_solution.isChecked() is True:
            self.results.lda.plot_solutionvector(alpha=alpha)
        if self.cb_lda_lcruve.isChecked() is True:
            self.results.lda.plot_lcurve()
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
