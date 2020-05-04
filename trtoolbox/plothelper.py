import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as colors


class PlotHelper():
    """ Object for interactive plotting. This class ensures that
    all the matplotlib objects are kept referenced which ensures
    proper function of the sliders.

    Attributes
    ----------
    fig_traces : matplotlib.figure.Figure
    ax_traces : matplotlib.axes.Axes
    l1_traces : matplotlib.lines.Line2D
    l2_traces: matplotlib.lines.Line2D
    axfreq : matplotlib.axes.Axes
    sfreq : matplotlib.widgets.Slider
    fig_spectra : matplotlib.figure.Figure
    ax_spectra : matplotlib.axes.Axes
    l1_spectra : matplotlib.lines.Line2D
    l2_spectra : matplotlib.lines.Line2D
    axtime : matplotlib.axes.Axes
    stime : matplotlib.widgets.Slider
    fig_lda = matplotlib.figure.Figure()
    axs_lda = np.array
    map_lda = matplotlib.contour.QuadContourSet
    ldadata = matplotlib.collections.QuadMesh
    l1_lda = matplotlib.lines.Line2D
    axalpha = matplotlib.axes.Axes
    salpha = matplotlib.widgets.Slider
    axcolor : str
    """

    def __init__(self):
        self.fig_traces = matplotlib.figure.Figure()
        self.ax_traces = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.l1_traces = matplotlib.lines.Line2D([], [])
        self.l2_traces = matplotlib.lines.Line2D([], [])
        self.axfreq = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.sfreq = matplotlib.widgets.Slider(self.axfreq, '', 0, 1)

        self.fig_spectra = matplotlib.figure.Figure()
        self.ax_spectra = matplotlib.axes.Axes(self.fig_spectra, [0, 0, 0, 0])
        self.l1_spectra = matplotlib.lines.Line2D([], [])
        self.l2_spectra = matplotlib.lines.Line2D([], [])
        self.axtime = matplotlib.axes.Axes(self.fig_spectra, [0, 0, 0, 0])
        self.stime = matplotlib.widgets.Slider(self.axtime, '', 0, 1)

        self.fig_lda = matplotlib.figure.Figure()
        self.axs_lda = np.ndarray((2, 2))
        # self.map_lda = matplotlib.contour.QuadContourSet(
        #     matplotlib.axes.Axes(self.fig_lda, [0, 0, 0, 0])
        # )
        # self.ldadata = matplotlib.collections.QuadMesh()
        self.l1_lda = matplotlib.lines.Line2D([], [])
        self.axalpha = matplotlib.axes.Axes(self.fig_lda, [0, 0, 0, 0])
        self.salpha = matplotlib.widgets.Slider(self.axalpha, '', 0, 1)

        self.axcolor = 'lightgoldenrodyellow'

    def plot_heatmap(self, data, time, wn, title='data', newfig=True):
        """ Plots a nice looking heatmap.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.
        time : np.array
            Time array.
        wn : np.array
            Frequency array.
        title : np.array
            Title of plot. Default *data*.
        newfig : boolean
            Setting to False prevents the creation of a new figure.

        Returns
        -------
        nothing
        """

        if newfig is True:
            plt.figure()
        # ensuring that time spans columns
        if data.shape[1] != time.size:
            data = np.transpose(data)

        if time.size == 0 or wn.size == 0:
            pc = plt.pcolormesh(data, cmap='jet', shading='gouraud')
        else:
            pc = plt.pcolormesh(
                time,
                wn,
                data,
                cmap='jet',
                shading='gouraud',
                norm=MidpointNormalize(midpoint=0),
                )
        plt.xscale('log')
        plt.title(title)
        return pc

    def plot_contourmap(self, data, time, wn, title='data', newfig=True):
        """ Plots a nice looking contourmap.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.
        time : np.array
            Time array.
        wn : np.array
            Frequency array.
        title : np.array
            Title of plot. Default *data*.
        newfig : boolean
            Setting to False prevents the creation of a new figure.

        Returns
        -------
        nothing
        """

        if newfig is True:
            plt.figure()
        # ensuring that time spans columns
        if data.shape[1] != time.size:
            data = np.transpose(data)

        levels = 12
        if time.size == 0 or wn.size == 0:
            pc = plt.contourf(
                data,
                levels=levels,
                cmap='bwr',
                norm=MidpointNormalize(midpoint=0)
            )
            pc = plt.contour(
                data,
                levels=levels,
                linewidths=0.3,
                colors='k'
            )
        else:
            pc = plt.contourf(
                time.flatten(),
                wn.flatten(),
                data,
                levels=levels,
                cmap='bwr',
                norm=MidpointNormalize(midpoint=0)
            )
            pc = plt.contour(
                time.flatten(),
                wn.flatten(),
                data,
                levels=levels,
                linewidths=0.3,
                colors='k'
            )
        plt.xscale('log')
        plt.title(title)
        return pc

    def plot_traces(self, res, index_alpha=-1, alpha=-1):
        """ Plots interactive time traces.

        Parameters
        -------
        res : mysvd.Results
            Contains the data to be plotted.

        Returns
        -------
        nothing
        """

        if res.type == 'svd':
            title = 'Time traces\nblue: Raw, red: SVD'
            procdata = res.svddata
        elif res.type == 'lda':
            if res.method == 'tik':
                index_alpha, alpha = res.get_alpha(index_alpha, alpha)
                procdata = res.fitdata[:, :, index_alpha]
                title = 'Time traces\nblue: Raw, red: LDA (alpha=%.2f)' % (alpha)
            elif res.method == 'tsvd':
                procdata = res.fitdata
                title = 'Time traces\nblue: Raw, red: LDA (TSVD truncated at %i)' % (res.k)
        elif res.type == 'gf':
            title = 'Time traces\nblue: Raw, red: Global Fit'
            procdata = res.fitdata

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2)
        plt.plot([min(res.time), max(res.time)], [0, 0], '--', color='k')
        l1, = plt.plot(res.time.T, res.data[0, :])
        l2, = plt.plot(res.time.T, procdata[0, :])
        plt.xscale('log')
        ax.margins(x=0)

        axfreq = plt.axes([0.175, 0.05, 0.65, 0.03], facecolor=self.axcolor)
        sfreq = Slider(
            axfreq, res.wn_name,
            np.min(res.wn),
            np.max(res.wn),
            valinit=np.min(res.wn),
            valstep=abs(res.wn[1]-res.wn[0])
            )
        ymin = np.min(res.data[:, :])
        ymax = np.max(res.data[:, :])
        sc = 1.05
        ax.set_ylim(ymin*sc, ymax*sc)

        # TODO: check for missing values! may be take from traces
        def update(val):
            val = sfreq.val
            ind = np.where(res.wn == val)[0][0]
            l1.set_ydata(res.data[ind, :])
            l2.set_ydata(procdata[ind, :])
            # ymin = min(res.data[ind, :])
            # ymax = max(res.data[ind, :])
            # sc = 1.1
            # ax.set_ylim(ymin*sc, ymax*sc)

        sfreq.on_changed(update)

        self.fig_traces = fig
        self.ax_traces = ax
        self.l1_traces = l1
        self.l2_traces = l2
        self.axfreq = axfreq
        self.sfreq = sfreq

    def plot_spectra(self, res, index_alpha=-1, alpha=-1):
        """ Plots interactive spectra.

        Parameters
        -------
        res : mysvd.Results
            Contains the data to be plotted.

        Returns
        -------
        nothing
        """

        if res.type == 'svd':
            title = 'Spectra\nblue: Raw, red: SVD'
            procdata = res.svddata
        elif res.type == 'lda':
            if res.method == 'tik':
                index_alpha, alpha = res.get_alpha(index_alpha, alpha)
                procdata = res.fitdata[:, :, index_alpha]
                title = 'Spectra\nblue: Raw, red: LDA (alpha=%.2f)' % (alpha)
            elif res.method == 'tsvd':
                procdata = res.fitdata
                title = 'Spectra\nblue: Raw, red: LDA (TSVD truncated at %i)' % (res.k)
        elif res.type == 'gf':
            title = 'Spectra\nblue: Raw, red: Global Fit'
            procdata = res.fitdata

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2)
        plt.plot([min(res.wn), max(res.wn)], [0, 0], '--', color='k')
        l1, = plt.plot(res.wn, res.data[:, 0], 'o-', markersize=4)
        l2, = plt.plot(res.wn, procdata[:, 0], 'o-', markersize=4)
        ax.margins(x=0)

        # TODO: better init for valstep!
        axtime = plt.axes([0.175, 0.05, 0.65, 0.03], facecolor=self.axcolor)
        stime = Slider(
            axtime, res.time_name,
            np.log10(np.min(res.time)),
            np.log10(np.max(res.time)),
            valinit=np.log10(np.min(res.time[0])),
            valstep=0.01
            )
        stime.valtext.set_text('%1.2e' % (10**stime.val))
        ymin = np.min(res.data[:, :])
        ymax = np.max(res.data[:, :])
        sc = 1.05
        ax.set_ylim(ymin*sc, ymax*sc)

        def update(val):
            val = 10**(stime.val)
            ind = abs(val - res.time).argmin()
            stime.valtext.set_text('%1.2e' % (res.time[ind]))
            l1.set_ydata(res.data[:, ind])
            l2.set_ydata(procdata[:, ind])
            # ymin = min(res.data[:, ind])
            # ymax = max(res.data[:, ind])
            # sc = 1.1
            # ax.set_ylim(ymin*sc, ymax*sc)

        stime.on_changed(update)

        self.fig_spectra = fig
        self.ax_spectra = ax
        self.l1_spectra = l1
        self.l2_spectra = l2
        self.axtime = axtime
        self.stime = stime

    def append_ldamap(self, res, index_alpha=-1):
        """ Appends NaN values in order to expand the taus array
            to match the time array span.

        Parameters
        -------
        res : mylda.Results
            Contains the results to be plotted.
        index_alpha : int
            Index of selected alpha value.

        Returns
        -------
        x_k : np.array
            LDA map of selected alpha value.
        taus : np.array
            Extended taus array.
        """

        if index_alpha > 0:
            x_k = res.x_k[:, :, index_alpha]
        elif index_alpha == -1:
            x_k = res.x_k
        nanarray = np.empty(np.shape(res.wn))
        nanarray[:] = np.NaN
        nanarray = np.hstack((nanarray, nanarray))
        if np.min(res.time) < np.min(res.taus):
            taus = np.insert(
                res.taus,
                0,
                [np.min(res.time), np.min(res.taus)*0.99]
            )
            x_k = np.hstack((nanarray, x_k))
        if np.max(res.time) > np.max(res.taus):
            taus = np.append(taus, [np.max(res.taus)*1.01, np.max(res.time)])
            taus = taus.reshape((1, taus.size))
            x_k = np.hstack((x_k, nanarray))
        # print(x_k)
        return x_k, taus

    def plot_ldaresults(self, res):
        """ Plots interactive resaults of LDA.

        Parameters
        -------
        res : mylda.Results
            Contains the results to be plotted.

        Returns
        -------
        nothing
        """

        fig, axs = plt.subplots(2, 2, figsize=[6.4*2, 4.8*2])

        # original data
        plt.sca(axs[0, 0])
        self.plot_heatmap(
            res.data, res.time, res.wn,
            title='Original Data', newfig=False)
        plt.ylabel('%s / %s' % (res.wn_name, res.wn_unit))
        plt.xlabel('%s / %s' % (res.time_name, res.time_unit))

        # plot lda data
        if res.method == 'tik':
            index_alpha = int(np.ceil(res.alphas.size/2))
            ldadata = np.transpose(res.dmatrix.dot(res.x_k[:, :, index_alpha].T))
        elif res.method == 'tsvd':
            ldadata = np.transpose(res.dmatrix.dot(res.x_k.T))
        plt.sca(axs[0, 1])
        pc_ldadata = self.plot_heatmap(
            ldadata, res.time, res.wn,
            title='LDA data', newfig=False)
        plt.ylabel('%s / %s' % (res.wn_name, res.wn_unit))
        plt.xlabel('%s / %s' % (res.time_name, res.time_unit))

        # lda map
        plt.sca(axs[1, 1])
        if res.method == 'tik':
            x_k, taus = self.append_ldamap(res, index_alpha)
        elif res.method == 'tsvd':
            x_k, taus = self.append_ldamap(res)
        pc_map = self.plot_contourmap(
            x_k, taus, res.wn,
            title='LDA Map', newfig=False)
        plt.ylabel('%s / %s' % (res.wn_name, res.wn_unit))
        plt.xlabel('%s / %s' % ('tau', res.time_unit))

        # lcurve
        if res.method == 'tik':
            plt.sca(axs[1, 0])
            plt.plot(res.lcurve[:, 0], res.lcurve[:, 1], 'o-', markersize=2)
            l1, = plt.plot(
                res.lcurve[index_alpha, 0],
                res.lcurve[index_alpha, 1],
                'o-',
                markersize=4,
                color='r'
            )
            plt.title('L-curve')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.125)

            axalpha = plt.axes([0.175, 0.03, 0.65, 0.02], facecolor=self.axcolor)
            salpha = Slider(
                axalpha, 'Alpha',
                np.log10(np.min(res.alphas)),
                np.log10(np.max(res.alphas)),
                valinit=np.log10(np.min(res.alphas[index_alpha])),
                valstep=abs(np.log10(res.alphas[1]) - np.log10(res.alphas[0]))
                )
            salpha.valtext.set_text('%1.2e' % (10**salpha.val))

            def update(val):
                val = 10**(salpha.val)
                ind = abs(val - res.alphas).argmin()
                salpha.valtext.set_text('%1.2e' % (res.alphas[ind]))
                # lda map
                plt.sca(axs[1, 1])
                plt.cla()
                x_k, _ = self.append_ldamap(res, ind)
                self.plot_contourmap(
                    x_k, taus, res.wn,
                    title='LDA Map', newfig=False)
                plt.ylabel('%s / %s' % (res.wn_name, res.wn_unit))
                plt.xlabel('%s / %s' % ('tau', res.time_unit))
                # lda data
                ldadata = np.transpose(res.dmatrix.dot(res.x_k[:, :, ind].T))
                pc_ldadata.set_array(ldadata.ravel())
                # lcurve
                l1.set_data(res.lcurve[ind, 0], res.lcurve[ind, 1])

            salpha.on_changed(update)
            self.l1_lda = l1
            self.axalpha = axalpha
            self.salpha = salpha

        self.fig_lda = fig
        self.axs_lda = axs
        self.map_lda = pc_map
        self.ldadata = pc_ldadata


class MidpointNormalize(colors.Normalize):
    """ Class for setting 0 as midpoint in the colormap.

    Attributes
    ----------
    midpoint : float
        Midpoint of colormap
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
