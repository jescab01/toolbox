
import time
import math
import numpy as np
from collections import Counter

from scipy import signal, fft
from mne import filter, time_frequency

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
import plotly.express as px

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from pyToolbox.signals import epochingTool
from pyToolbox import mnetools


## Static functional connetivity
def fc(signals, samplingFreq=None, lowcut=8, highcut=12, measure="PLV", ef=None, regionLabels=None,
       folder=None, plot=None, verbose=False, auto_open=False):
    """

    """
    tic = time.time()

    n_rois = len(signals)
    fc_matrix = np.ndarray((n_rois, n_rois))

    if "CORR" in measure:
        # stdSignals = (signals - np.average(signals, axis=0)) / np.std(signals, axis=0)

        # normalSignals = np.ndarray((len(signals), len(signals[0])))
        # for channel in range(len(signals)):
        #     mean = np.mean(signals[channel, :])
        #     std = np.std(signals[channel, :])
        #     normalSignals[channel] = (signals[channel, :] - mean) / std

        print("Calculating CORR", end="") if verbose else None
        signals = (signals - np.mean(signals, axis=1, keepdims=True)) / np.std(signals, axis=1, keepdims=True)
        fc_matrix = np.corrcoef(signals)

        # for r1, roi1 in enumerate(range(n_rois)):
        #     print(" . . . %0.2f %%" % ((r1+1)/n_rois), end="\r") if verbose else None
        #     for roi2 in range(n_rois):
        #         fc_matrix[roi1][roi2] = np.corrcoef(signals[roi1], signals[roi2])[0, 1]
        #         # fc_matrix[roi1][roi2] = sum(signals[roi1] * signals[roi2]) / len(signals[0])

    elif measure in ["PLV", "AEC", "PLI"]:

        if not ef:
            filterSignals = filter.filter_data(signals, samplingFreq, lowcut, highcut, verbose=verbose)
            efSignals = epochingTool(filterSignals, 4, 4, samplingFreq, "signals", verbose=verbose)

            # Obtain Analytical signal
            efPhase, efEnvelope = [], []
            for i in range(len(efSignals)):
                analyticalSignal = signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.angle(analyticalSignal))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)
        elif "efPhase" in ef:
            efPhase = np.array(signals)[np.newaxis, :, :]
        elif "efEnvelope" in ef:
            efEnvelope = np.array(signals)[np.newaxis, :, :]

        if "PLV" in measure:
            print("Calculating PLV", end="") if verbose else None
            for r1, roi1 in enumerate(range(n_rois)):
                print(" . . . %0.2f %%" % ((r1 + 1) / n_rois), end="\r") if verbose else None
                for roi2 in range(n_rois):
                    plv_values = list()
                    for epoch in range(len(efPhase)):
                        phaseDifference = efPhase[epoch][roi1] - efPhase[epoch][roi2]
                        value = abs(np.average(np.exp(1j * phaseDifference)))
                        plv_values.append(value)
                    fc_matrix[roi1, roi2] = np.average(plv_values)

        elif "AEC" in measure:
            print("Calculating AEC", end="") if verbose else None
            for r1, roi1 in enumerate(range(n_rois)):
                print(" . . . %0.2f %%" % ((r1 + 1) / n_rois), end="\r") if verbose else None
                for roi2 in range(n_rois):
                    values_aec = list()  # AEC per epoch and channel x channel
                    for epoch in range(len(efEnvelope)):  # CORR between channels by epoch
                        r = np.corrcoef(efEnvelope[epoch][roi1], efEnvelope[epoch][roi2])
                        values_aec.append(r[0, 1])
                    fc_matrix[roi1, roi2] = np.average(values_aec)

        elif "PLI" in measure:
            print("Calculating PLI", end="") if verbose else None
            for r1, roi1 in enumerate(range(n_rois)):
                print(" . . . %0.2f %%" % ((r1 + 1) / n_rois), end="\r") if verbose else None
                for roi2 in range(n_rois):
                    pli_values = list()
                    for epoch in range(len(efPhase)):
                        phaseDifference = efPhase[epoch][roi1] - efPhase[epoch][roi2]
                        value = np.abs(np.average(np.sign(np.sin(phaseDifference))))
                        pli_values.append(value)
                    fc_matrix[roi1, roi2] = np.average(pli_values)

    else:
        print("Unkown measure. Exit")

    if plot:
        fig = go.Figure(data=go.Heatmap(z=fc_matrix, x=regionLabels, y=regionLabels,
                                        colorbar=dict(thickness=4), colorscale='Viridis'))
        fig.update_layout(title='Phase Locking Value', height=500, width=500)

        if plot == "html":
            pio.write_html(fig, file=folder + "/" + measure + ".html", auto_open=auto_open)
        elif plot == "png":
            pio.write_image(fig, file=folder + "/" + measure + ".png", engine="kaleido")
        elif plot == "svg":
            pio.write_image(fig, file=folder + "/" + measure + ".svg", engine="kaleido")
        elif plot == "inline":
            plotly.offline.iplot(fig)

    print("  -  %0.3f seconds.\n" % (time.time() - tic,)) if verbose else None

    return fc_matrix


## Dynamical functional connetivity
def dynamic_fc(data, samplingFreq, transient, window, step, measure="PLV", plot=None, folder='figures',
               lowcut=8, highcut=12, filtered=False, auto_open=False, verbose=False, mode="dFC"):
    """
    Calculates dynamical Functional Connectivity using the classical method of sliding windows.

    REFERENCE || Cabral et al. (2017) Functional connectivity dynamically evolves on multiple time-scales over a
    static structural connectome: Models and mechanisms

    :param data: Signals in shape [ROIS x time]
    :param samplingFreq: sampling frequency (Hz)
    :param window: Seconds of sliding window
    :param step: Movement step for sliding window
    :param measure: FC measure (PLV; AEC)
    :param plot: Plot dFC matrix?
    :param folder: To save figures output
    :param auto_open: on browser.
    :return: dFC matrix
    """

    window_ = window * 1000
    step_ = step * 1000

    if len(data[0]) > window_:
        if filtered:
            filterSignals = data
        else:
            # Band-pass filtering
            filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=verbose)
        if verbose:
            print("Calculating dFC matrix...")

        matrices_fc = list()
        for w in np.arange(0, (len(data[0])) - window_, step_, 'int'):

            if verbose:
                print('%s %i / %i' % (measure, w / step_, ((len(data[0])) - window_) / step_))

            efSignals = filterSignals[:, w:w + window_]

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            # # Check point
            # from toolbox.signals import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(data, raw_time, regionLabels)
            # plotConversions(data[:, :len(efSignals[0])], efSignals, efPhase, efEnvelope, band="alpha", regionLabels=regionLabels)

            efData, ef = (efPhase, "efPhase") if measure in ["PLV", "PLI"] else (efEnvelope, "efEnvelope")
            matrices_fc.append(fc(efData, ef=ef, measure=measure, verbose=verbose))


        dFC_matrix = np.zeros((len(matrices_fc), len(matrices_fc)))
        for t1 in range(len(matrices_fc)):
            for t2 in range(len(matrices_fc)):
                dFC_matrix[t1, t2] = np.corrcoef(matrices_fc[t1][np.triu_indices(len(matrices_fc[0]), 1)],
                                                 matrices_fc[t2][np.triu_indices(len(matrices_fc[0]), 1)])[1, 0]

        if plot:
            fig = go.Figure(
                data=go.Heatmap(z=dFC_matrix, x=np.arange(transient, transient + len(data[0]), step_) / 1000,
                                y=np.arange(transient, transient + len(data[0]), step_) / 1000,
                                colorscale='Viridis', colorbar=dict(thickness=4)))
            fig.update_layout(title='dynamical Functional Connectivity', height=400, width=400)
            fig.update_xaxes(title="Time 1 (seconds)")
            fig.update_yaxes(title="Time 2 (seconds)")

            if plot == "html":
                pio.write_html(fig, file=folder + "/dPLV.html", auto_open=auto_open)
            elif plot == "png":
                pio.write_image(fig, file=folder + "/dPLV_" + str(time.time()) + ".png", engine="kaleido")
            elif plot == "svg":
                pio.write_image(fig, file=folder + "/dPLV.svg", engine="kaleido")
            elif plot == "inline":
                plotly.offline.iplot(fig)

        if mode == "all_matrices":
            return dFC_matrix, matrices_fc
        else:
            return dFC_matrix

    else:
        print('Error: Signal length should be longer than window length (%i sec)' % window)


def ple(efPhase, time_lag, pattern_size, samplingFreq, subsampling=1):
    """
    It calculates Phase Lag Entropy (Lee et al., 2017) on a bunch of filtered and epoched signals with shape [epoch,rois,time]
    It is based on the diversity of temporal patterns between two signals phases.

    A pattern S(t) is defined as:
        S(t)={s(t), s(t+tau), s(t+2tau),..., s(t + m*tau -1)} where
                (s(t)=1 if delta(Phi)>0) & (s(t)=0 if delta(Phi)<0)

    PLE = - sum( p_j * log(p_j) ) / log(2^m)

        where
        - p_j is the probability of the jth pattern, estimated counting the number of times each pattern
        occurs in a given epoch and
        - m is pattern size.

    REFERENCE ||  Lee et al. (2017) Diversity of FC patterns is reduced during anesthesia.


    :param efPhase: Phase component of Hilbert transform (filtered in specific band and epoched)
    :param time_lag: "tau" temporal distance between elements in pattern
    :param pattern_size: "m" number of elements to consider in each pattern
    :param samplingFreq: signal sampling frequency
    :param subsampling: If your signal has high temporal resolution, maybe gathering all possible patters is not
     efficient, thus you can omit some timepoints between gathered patterns

    :return: PLE - matrix shape (rois, rois) with PLE values for each couple; patts - patterns i.e. all the patterns
    registered in the signal and the number of times they appeared.
    """

    tic = time.time()
    try:
        efPhase[0][0][0]  # Test whether signals have been epoched
        PLE = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        time_lag = int(np.trunc(time_lag * samplingFreq / 1000))  # translate time lag in timepoints
        patts = list()
        print("Calculating PLE ", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                ple_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]

                    # Binarize to create the pattern and comprehend into list
                    patterns = [str(np.where(phaseDifference[t: t + time_lag * pattern_size: time_lag] > 0, 1, 0)) for t
                                in
                                np.arange(0, len(phaseDifference) - time_lag * pattern_size, step=subsampling)]
                    patt_counts = Counter(patterns)
                    patts.append(patt_counts)
                    summation = 0
                    for key, value in patt_counts.items():
                        p = value / len(patterns)
                        summation += p * np.log10(p)

                    ple_values.append((-1 / np.log10(2 ** pattern_size)) * summation)
                PLE[channel1, channel2] = np.average(ple_values)
        print("%0.3f seconds.\n" % (time.time() - tic,))

        return PLE, patts

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


## Metastability
def kuramoto_order(data, samplingFreq, lowcut=8, highcut=12, filtered=False, verbose=False):
    """
    From Deco et al. (2017) The dynamics of resting fluctuations in the brain: metastability and its dynamical cortical core
    "We measure the metastability as the standard deviation of the Kuramoto order parameter across time".
    """
    if verbose:
        print("Calculating Kuramoto order paramter...")
    if filtered:
        filterSignals = data

    else:
        # Band-pass filtering
        filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=verbose)

    # Padding as Hilbert transform has distortions at edges
    padding = np.zeros((len(data), 1000))

    filterSignals_padded = np.concatenate([padding, filterSignals, padding], axis=1)

    analyticalSignal_padded = signal.hilbert(filterSignals_padded)
    # Get instantaneous phase by channel
    efPhase_padded = np.angle(analyticalSignal_padded)

    efPhase = efPhase_padded[:, 1000:-1000]

    # Kuramoto order parameter in time
    kuramoto_array = abs(np.sum(np.exp(1j * efPhase), axis=0)) / len(efPhase)
    # Average Kuramoto order parameter for the set of signals
    kuramoto_avg = np.average(kuramoto_array)
    kuramoto_sd = np.std(kuramoto_array)

    return kuramoto_array, kuramoto_sd, kuramoto_avg


def kuramoto_polar(data, time_, samplingFreq, speed, lowcut=8, highcut=10, timescale="ms",
                   mode="html", folder="figures", title="", auto_open=True):
    # Band-pass filtering
    filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=False)

    # Padding as Hilbert transform has distortions at edges
    padding = np.zeros((len(data), 1000))

    filterSignals_padded = np.concatenate([padding, filterSignals, padding], axis=1)

    analyticalSignal_padded = signal.hilbert(filterSignals_padded)
    # Get instantaneous phase by channel
    efPhase_padded = np.angle(analyticalSignal_padded)

    phases = efPhase_padded[:, 1000:-1000]

    phases = phases[:, ::speed]
    time_ = time_[::speed]

    kuramoto_order = [1 / len(phases) * np.sum(np.exp(1j * phases[:, i])) for i, dp in enumerate(phases[0])]
    KO_magnitude = np.abs(kuramoto_order)
    KO_angle = np.angle(kuramoto_order)

    wraped_phase = (phases % (2 * np.pi))
    cmap = (px.colors.qualitative.Plotly + px.colors.qualitative.Light24 + px.colors.qualitative.Set2 +
            px.colors.qualitative.Dark24 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel2 +
            px.colors.qualitative.Set3 + px.colors.qualitative.Light24 + px.colors.qualitative.Pastel1 +
            px.colors.qualitative.Alphabet + px.colors.qualitative.Vivid + px.colors.qualitative.Antique +
            px.colors.qualitative.Safe + px.colors.qualitative.D3 + px.colors.qualitative.Prism +
            px.colors.qualitative.Bold + px.colors.qualitative.G10) * 50

    # With points
    fig = go.Figure()

    ## Add Kuramoto Order
    fig.add_trace(go.Scatterpolar(theta=[KO_angle[0]], r=[KO_magnitude[0]], thetaunit="radians",
                                  name="KO", mode="markers", marker=dict(size=6, color="darkslategray")))

    ## Add each region phase
    fig.add_trace(go.Scatterpolar(theta=wraped_phase[:, 0], r=[1] * len(wraped_phase), thetaunit="radians",
                                  name="ROIs", mode="markers", marker=dict(size=8, color=cmap), opacity=0.8))

    fig.update(frames=[go.Frame(data=[go.Scatterpolar(theta=[KO_angle[i]], r=[KO_magnitude[i]]),
                                      go.Scatterpolar(theta=wraped_phase[:, i])],
                                traces=[0, 1], name=str(np.round(t, 3))) for i, t in enumerate(time_)])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(template="plotly_white", height=400, width=500, polar=dict(angularaxis_thetaunit="radians", ),

                      sliders=[dict(
                          steps=[
                              dict(method='animate',
                                   args=[[str(t)], dict(mode="immediate",
                                                        frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=0))], label=str(np.round(t, 3))) for
                              i, t in enumerate(time_)],
                          transition=dict(duration=0), xanchor="left", x=0.35, y=-0.15,
                          currentvalue=dict(font=dict(size=15, color="black"), prefix="Time (%s) - " % (timescale),
                                            visible=True,
                                            xanchor="right"),
                          len=0.7, tickcolor="white", font=dict(color="white"))],

                      updatemenus=[
                          dict(type="buttons", showactive=False, x=0.05, y=-0.4, xanchor="left", direction="left",
                               buttons=[
                                   dict(label="\u23f5", method="animate",
                                        args=[None,
                                              dict(frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                   transition=dict(duration=0),
                                                   fromcurrent=True, mode='immediate')]),
                                   dict(label="\u23f8", method="animate",
                                        args=[[None],
                                              dict(frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                   transition=dict(duration=0),
                                                   mode="immediate")])])])

    if "html" in mode:
        pio.write_html(fig, file=folder + "/Animated_PolarKuramoto-" + title + ".html", auto_open=auto_open,
                       auto_play=False)

    elif "inline" in mode:
        plotly.offline.iplot(fig)


"""
Following functions were created on Tue Feb  7 13:29:32 2023
@author: Ricardo Bruña

Efficient implmentation of PLV, COH and AEC, with source leakage corrected versions.


Edited on 08/10/24 by @Jescab01.
"""

def plv(data, band=None, padding=None, average=True):
    # Checks whether the data is a valid MNE object.
    # For now, it only works with sensor-space data.
    if not (isinstance(data, mnetools.mnevalid)):
        raise TypeError('Unsupported data type.')

    # Checks the input.
    if (band is None) and np.isreal(data._data).all():
        raise TypeError('No filter provided.')

    # Makes a copy of the input to work with.
    data = data.copy()

    # Filters the data, if required.
    if band is not None:

        # Uses the provided padding, if any.
        if padding is not None:
            padding = padding * data.info['sfreq']
            padding = np.floor(padding).astype(int)

        # Otherwise tries to get the padding from the MNE object.
        elif hasattr(data, 'padding'):
            padding = data.padding * data.info['sfreq']
            padding = np.floor(padding).astype(int)

        # Otherwise assumes that the padding is the negative time.
        else:
            # print ( 'Assuming that padding is equal to negative time.' )
            padding = data.time_as_index(0)[0]

        # Defines the filter.
        num = signal.firwin(padding, band, fs=data.info['sfreq'], window='hamming', pass_zero='bandpass')

        # Filters the epoched data using Hilbert filtering.
        data = mnetools.filtfilt(data, num=num, hilbert=True)

        # Removes the padding.
        if padding and padding > 0:
            tmin = data.times[padding]
            tmax = data.times[-padding]
            data = data.crop(tmin=tmin, tmax=tmax, include_tmax=False)

    # Extracts the raw data.
    rawdata = data.get_data()

    # Gets the metadata.
    shape = rawdata.shape
    nsamp = rawdata.shape[-1]
    nchan = rawdata.shape[-2]

    # Reshapes as repetitions x channels x samples.
    rawdata = rawdata.reshape([-1, nchan, nsamp])
    nrep = rawdata.shape[0]

    # Normalizes the complex array.
    tiny = np.finfo(rawdata.dtype).tiny
    rawnorm = rawdata / (np.abs(rawdata) + tiny)

    # In the current implementation, the nodes are channels.
    nodes = data.ch_names
    nnode = nchan

    # Initializes the complex PLV matrix.
    cplv = np.zeros((nrep, nnode, nnode), dtype=np.complex128)

    # Gets the complex PLV by matrix multiplication.
    for i in range(nrep):
        cplv[i] = np.inner(rawnorm[i], rawnorm[i].conj()) / nsamp

    # Recovers the original shape of the data.
    cplv = cplv.reshape(shape[:-2] + (nnode, nnode))

    # Calculates the PLV and its corrected imaginary counterpart.
    plv = np.abs(cplv)
    iplv = np.imag(cplv)
    rplv = np.real(cplv)
    ciplv = abs(iplv / np.sqrt(np.maximum(1 - rplv * rplv, tiny)))

    # Forces the diagonal of ciPLV to 0.
    diag = np.diag_indices(nnode)
    ciplv[..., diag[0], diag[1]] = 0

    # # Flattens the connectivity matrices.
    # plv = plv.reshape((nrep, nnode * nnode))
    # ciplv = ciplv.reshape((nrep, nnode * nnode))

    # # Defines the upper diagonal in the flattened data.
    # triu = np.triu_indices(nnode, k=0)
    # triu = np.ravel_multi_index(triu, (nnode, nnode))
    #
    # # Keeps only the upper triangular.
    # plv = plv[:, triu]
    # ciplv = ciplv[:, triu]

    # # Creates the epoched connectivity objects.
    # plv = mne_connectivity.EpochConnectivity(
    #     data=plv,
    #     n_nodes=nnode,
    #     indices='symmetric',
    #     names=nodes,
    #     method='Temporal PLV',
    #     n_epochs_used=nrep)
    #
    # ciplv = mne_connectivity.EpochConnectivity(
    #     data=ciplv,
    #     n_nodes=nnode,
    #     indices='symmetric',
    #     names=nodes,
    #     method='Temporal ciPLV',
    #     n_epochs_used=nrep)

    # Averages the epochs connectivity, if requested.
    if average:
        plv = np.average(plv, axis=0)
        ciplv = np.average(ciplv, axis=0)

    # Returns the estimated connectivity.
    return plv, ciplv


def coh(data, band=None, padding=None, average=True, faverage=True):
    """
    Corrected imaginary part of coherence taken from:
    * Ewald et al. 2012 NeuroImage 60:476-488 Eq. 19.
    """

    # Checks whether the data is a valid MNE object.
    if not (isinstance(data, mnetools.mnevalid)):
        raise TypeError('Unsupported data type.')

    # Checks the input.
    if band is None:
        band = (0, np.inf)

    # Makes a copy of the input to work with.
    data = data.copy()

    # Uses the provided padding, if any.
    if padding is not None:
        padding = padding * data.info['sfreq']
        padding = np.floor(padding).astype(int)

    # Otherwise tries to get the padding from the MNE object.
    elif hasattr(data, 'padding'):
        padding = data.padding * data.info['sfreq']
        padding = np.floor(padding).astype(int)

    # Otherwise assumes that the padding is the negative time.
    else:
        # print ( 'Assuming that padding is equal to negative time.' )

        padding = data.time_as_index(0)[0]

    # Removes the padding, if requried.
    if (padding and padding > 0):
        tmin = data.times[padding]
        tmax = data.times[-padding]
        data = data.crop(tmin=tmin, tmax=tmax, include_tmax=False)

    # Extracts the raw data.
    rawdata = data.get_data()

    # Gets the metadata.
    shape = rawdata.shape
    nsamp = rawdata.shape[-1]
    nchan = rawdata.shape[-2]

    # Reshapes as repetitions x channels x samples.
    rawdata = rawdata.reshape([-1, nchan, nsamp])
    nrep = rawdata.shape[0]

    # Gets the default window length and overlap.
    winlen = int(nsamp / 9 * 2)
    overlap = int(winlen / 2)

    # Calculates the number of windows.
    nwin = int((nsamp - overlap) / (winlen - overlap))

    # In the current implementation, the nodes are channels.
    nodes = data.ch_names
    nnode = nchan

    # Initialzies the windowed data matrix.
    windata = np.zeros((nwin, nrep, nnode, winlen))

    # Goes through each window.
    for index in range(nwin):
        # Calculates the window offset in the epoch.
        offset = index * (winlen - overlap)

        # Gets the data.
        windata[index] = rawdata[None, ..., offset: offset + winlen]

    # Applies the tapper.
    windata = windata * signal.hamming(winlen)

    # Calculates the size of the Fourier transform.
    nfft = int(2 ** np.ceil(np.log2(winlen)))
    nfft = np.max((nfft, 256))

    # Calculates the Fourier transform of the data.
    fdata = fft.fft(windata, n=nfft, axis=-1, workers=-1)

    # Keeps only the desired part of the spectrum.
    freqs = fft.fftfreq(nfft, 1 / data.info['sfreq'])
    findex = (band[0] <= freqs) & (freqs <= band[1])
    fdata = fdata[..., findex]
    freqs = freqs[findex]
    nfreq = fdata.shape[-1]

    # Gets the per-window cross-spectra.
    cross = fdata[..., None, :] * fdata[..., None, :, :].conj()

    # Gets the average cross- and auto-spectra.
    mcross = np.mean(cross, axis=0)
    mauto = mcross[..., range(nnode), range(nnode), :]

    # Gets the coherency values.
    num = mcross
    den = np.sqrt(mauto[:, None, :, :] * mauto[:, :, None, :])
    coh = num / den

    # Recovers the original shape of the data.
    coh = coh.reshape(shape[:-2] + (nnode, nnode, -1))

    # Calculates the magnitude-squared coherence and its corrected imaginary counterpart.
    tiny = np.finfo(rawdata.dtype).tiny
    mscoh = np.abs(coh) ** 2
    icoh = np.imag(coh)
    rcoh = np.real(coh)
    cicoh = abs(icoh / np.sqrt(np.maximum(1 - rcoh * rcoh, tiny)))

    # Forces the diagonal of corrected imaginary coherence to 0.
    diag = np.diag_indices(nnode)
    cicoh[..., diag[0], diag[1], :] = 0

    # Averages across frequencies, if requested.
    if faverage:
        mscoh = mscoh.mean(axis=-1, keepdims=True)
        cicoh = cicoh.mean(axis=-1, keepdims=True)
        freqs = freqs.mean(axis=0, keepdims=True)
        nfreq = 1

    # # Flattens the connectivity matrix.
    # mscoh = mscoh.reshape((nrep, nnode * nnode, nfreq))
    # cicoh = cicoh.reshape((nrep, nnode * nnode, nfreq))

    # # Defines the upper diagonal in the flattened data.
    # triu = np.triu_indices(nnode, k=0)
    # triu = np.ravel_multi_index(triu, (nnode, nnode))
    #
    # # Keeps only the upper triangular.
    # mscoh = mscoh[:, triu, 0]
    # cicoh = cicoh[:, triu, 0]

    # # Creates the epoched connectivity objects.
    # mscoh = mne_connectivity.EpochSpectralConnectivity(
    #     data=mscoh,
    #     n_nodes=nnode,
    #     freqs=freqs,
    #     indices='symmetric',
    #     names=nodes,
    #     method='Magnitude-squared coherence',
    #     n_epochs_used=nrep)
    #
    # cicoh = mne_connectivity.EpochSpectralConnectivity(
    #     data=cicoh,
    #     n_nodes=nnode,
    #     freqs=freqs,
    #     indices='symmetric',
    #     names=nodes,
    #     method='Corrected imaginary part of coherence',
    #     n_epochs_used=nrep)

    # Averages the epochs connectivity, if requested.
    if average:
        mscoh = np.average(mscoh, axis=0)
        cicoh = np.average(cicoh, axis=0)

    # Returns the estimated connectivity.
    return mscoh, cicoh


def aec(data, ortho=False, band=None, decimate=False, padding=None, smoothing=0, continuous=False, average=True,
        single=False):
    # Checks whether the data is a valid MNE object.
    # For now, it only works with sensor-space data.
    if not (isinstance(data, mnetools.mnevalid)):
        raise TypeError('Unsupported data type.')

    # Checks the input.
    if (band is None) and np.isreal(data._data).all():
        raise TypeError('No filter provided.')

    # Makes a copy of the input to work with.
    data = data.copy()

    # Filters the data, if required.
    if band is not None:

        # Uses the provided padding, if any.
        if padding is not None:
            padding = padding * data.info['sfreq']
            padding = np.floor(padding).astype(int)

        # Otherwise tries to get the padding from the MNE object.
        elif hasattr(data, 'padding'):
            padding = data.padding * data.info['sfreq']
            padding = np.floor(padding).astype(int)

        # Otherwise assumes that the padding is the negative time.
        else:
            # print ( 'Assuming that padding is equal to negative time.' )

            padding = data.time_as_index(0)[0]

        # Defines the filter.
        num = signal.firwin(padding, band, fs=data.info['sfreq'], window='hamming', pass_zero='bandpass')

        # Filters the epoched data using Hilbert filtering.
        data = mnetools.filtfilt(data, num=num, hilbert=True)

        '''
        # Removes the padding.
        if padding and padding > 0:
            tmin    = data.times [  padding ]
            tmax    = data.times [ -padding ]
            data    = data.crop ( tmin = tmin, tmax = tmax, include_tmax = False )
        '''

        # Decimates the data, if requested.
        if decimate:
            # Decimates the data to the optimal sampling rate (2.1 times the maximum frequency).
            ratio = np.floor(data.info['sfreq'] / (2.1 * band[1])).astype(int)
            data = mnetools.decimate(data, ratio)

            # Corrects the padding.
            padding = (padding / ratio).astype(int)

    # Extracts the raw data.
    rawdata = data.get_data()

    # Transforms the data into single precision, if requested.
    if single:
        rawdata = rawdata.astype(np.float32)

    # Defines the minimum possible value.
    tiny = np.finfo(rawdata.dtype).eps

    # Defines the padding and the smoothing in samples.
    spadd = padding
    ssmooth = round(smoothing * data.info['sfreq'])

    # Shortens the padding, if possible.
    # if spadd > math.ceil(ssmooth / 2):
    #     # Calculates the extra padding.
    #     xspadd = spadd - math.ceil(ssmooth / 2)
    #
    #     # Removes the extra padding.
    #     rawdata = rawdata[..., xspadd: -xspadd]
    #
    #     # Updates the value of the padding.
    #     spadd = math.ceil(ssmooth / 2)

    # Gets the metadata.
    nsamp = rawdata.shape[-1]
    nchan = rawdata.shape[-2]

    # In the current implementation, the nodes are channels.
    nodes = data.ch_names
    nnode = nchan

    # Rewrites as channels x repetitions x samples.
    rawdata = rawdata.reshape([-1, nnode, nsamp])
    rawdata = np.swapaxes(rawdata, 0, 1)
    nrep = rawdata.shape[1]

    # Initializes the AEC matrices.
    aecov = np.zeros((nnode, nnode, nrep))
    aecovlc = np.zeros((nnode, nnode, nrep))
    nreg = np.zeros((nnode, nnode, nrep))

    # Splits in real and imaginary parts.
    rawreal = np.real(rawdata)
    rawimag = np.imag(rawdata)

    # Gets the envelope of the signal.
    rawenv = np.abs(rawdata)

    # Smooths the envelope and removes the padding.
    # rawenv = auxaec.get_mas(rawenv, ssmooth, spadd)

    # Removes the padding.
    rawenv  = rawenv [..., padding : -padding ]

    # Centers the envelope (as continuous or per repetitions).
    if continuous:
        rawenv  = rawenv - rawenv.mean ( axis = -1, keepdims = True ).mean ( axis = -2, keepdims = True )
        # rawenv = auxaec.demean2(rawenv)
    else:
        rawenv  = rawenv - rawenv.mean ( axis = -1, keepdims = True )
        # rawenv = auxaec.demean(rawenv)

    # Gets the AE covariance by matrix multiplication.
    for irep in range(nrep):
        aecov[:, :, irep] = np.inner(rawenv[:, irep, :], rawenv[:, irep, :])

    # Gets the norm per node and repetition.
    dummy   = rawenv * rawenv
    nraw    = dummy.sum ( axis = -1 )
    # nraw = auxaec.get_norm(rawenv)

    # Otherwise orthogonalizes pairwise.
    if ortho is True:

        # Concatenates all the repetitions to estimate the projections.
        rawconc = rawreal[..., spadd: -spadd or None].reshape([nnode, -1])
        rawconc = rawconc - rawconc.mean(axis=-1, keepdims=True)

        # Calculates the betas of the regression.
        # proj    = np.inner ( rawconc, rawconc )
        proj = rawconc.dot(rawconc.T)
        betas = proj / np.diag(proj)

        # Goes through each signal.
        for inode in range(nnode):

            # Removes the projection of the current signal from all the others.
            regreal = rawreal - betas[:, [inode], None] * rawreal[[inode], :, :]
            regimag = rawimag - betas[:, [inode], None] * rawimag[[inode], :, :]
            # regreal = auxaec.get_lc ( rawreal, rawreal [ inode, :, : ], betas [ :, inode ] )
            # regimag = auxaec.get_lc ( rawimag, rawimag [ inode, :, : ], betas [ :, inode ] )

            # Gets the envelope.
            # regenv = auxaec.get_abs(regreal, regimag)
            regenv = np.sqrt(regreal ** 2 + regimag ** 2)

            # Smooths the envelope and removes the padding.
            # regenv = auxaec.get_mas(regenv, ssmooth, spadd)

            # Removes the padding.
            regenv  = regenv [..., spadd : -spadd ]

            # Centers the envelope (as continuous or per repetitions).
            if continuous:
                regenv  = regenv - regenv.mean ( axis = -1, keepdims = True ).mean ( axis = -2, keepdims = True )
                # regenv = auxaec.demean2(regenv)
            else:
                regenv  = regenv - regenv.mean ( axis = -1, keepdims = True )
                # regenv = auxaec.demean(regenv)

            # Gets the AECov per repetition by matrix multiplication.
            dummy   = rawenv [ [ inode ], :, : ] * regenv
            aecovlc [ inode, :, : ] = dummy.sum ( axis = -1 )
            # aecovlc[inode, :, :] = auxaec.get_dot(regenv, rawenv[inode, :, :])

            # Gets the norm per node and repetition.
            dummy   = regenv * regenv
            nreg [ inode, :, : ] = dummy.sum ( axis = -1 )
            # nreg[inode, :, :] = auxaec.get_norm(regenv)

    # If requested, simulates continuous data.
    if continuous:

        # Gets the covariance and norms of the continuous data.
        aecov = aecov.sum(axis=-1)
        aecovlc = aecovlc.sum(axis=-1)
        nraw = nraw.sum(axis=-1)
        nreg = nreg.sum(axis=-1)

        # Calculates the AEC of the continuous data as the normalized covariance.
        aec = aecov / (tiny * tiny + np.sqrt(nraw[:, None] * nraw))
        aeclc = aecovlc / (tiny * tiny + np.sqrt(nraw[:, None] * nreg))

        # Makes the leakage-corrected AEC symmetric.
        aeclc = (aeclc + aeclc.T) / 2

        # Sets the number of epochs to 1.
        nrep = 1

    # Otherwise treats each trial individually.
    else:

        # Calculates the AEC as the normalized covariance.
        aec = aecov / (tiny * tiny + np.sqrt(nraw[:, None, :] * nraw[None, :, :]))
        aeclc = aecovlc / (tiny * tiny + np.sqrt(nraw[:, None, :] * nreg))

        # Reshapes the AEC matrices as repetitions by nodes.
        aec = np.swapaxes(aec, -1, 0)
        aeclc = np.swapaxes(aeclc, -1, 0)

        # Makes the leakage-corrected AEC symmetric.
        aeclc = (aeclc + np.swapaxes(aeclc, -2, -1)) / 2

    # Forces the diagonal of the leakage-corrected AEC to be 0.
    diag = np.diag_indices(nnode)
    aeclc[..., diag[0], diag[1]] = 0

    # # Flattens the connectivity matrices.
    # aec = aec.reshape((nrep, nnode * nnode))
    # aeclc = aeclc.reshape((nrep, nnode * nnode))

    # # Defines the upper diagonal in the flattened data.
    # triu = np.triu_indices(nnode, k=0)
    # triu = np.ravel_multi_index(triu, (nnode, nnode))
    #
    # # Keeps only the upper triangular.
    # aec = aec[:, triu]
    # aeclc = aeclc[:, triu]

    # Averages the epochs connectivity, if requested.
    if average:
        aec = np.average(aec, axis=0)
        aeclc = np.average(aeclc, axis=0)

    # Returns the estimated connectivity.
    return aec, aeclc


def crossfreq_aec(data, bands=None, padding=None, continuous=False):

    # Checks whether the data is a valid MNE object.
    # For now, it only works with sensor-space data.
    if not (isinstance(data, mnetools.mnevalid)):
        raise TypeError('Unsupported data type.')

    # Checks the input.
    if (bands is None) and np.isreal(data._data).all():
        raise TypeError('No filter provided.')

    if not isinstance(bands, list):
        raise TypeError('You should probide a list of bands (tuples) for Cross Frequency analysis.')

    # Makes a copy of the input to work with.
    data = data.copy()

    # Uses the provided padding, if any.
    if padding is not None:
        padding = padding * data.info['sfreq']
        padding = np.floor(padding).astype(int)

    # Otherwise tries to get the padding from the MNE object.
    elif hasattr(data, 'padding'):
        padding = data.padding * data.info['sfreq']
        padding = np.floor(padding).astype(int)

    # Otherwise assumes that the padding is the negative time.
    else:
        # print ( 'Assuming that padding is equal to negative time.' )
        padding = data.time_as_index(0)[0]


    # Defines the filter.
    filters = [signal.firwin(padding, band, fs=data.info['sfreq'], window='hamming', pass_zero='bandpass') for band in bands]

    # Filters the epoched data using Hilbert filtering.
    data_filt = [mnetools.filtfilt(data, num=num, hilbert=True) for num in filters]

    # # CHECK - Plot raw and conversions
    # fig = go.Figure()
    # for i, signals_band in enumerate(data_filt):
    #
    #     ## For Raw input (signals x time)
    #     if len(signals_band.get_data().shape) == 2:
    #         hil = signals_band.get_data()[0, :]
    #         base = raw_data[0]-np.average(raw_data[0])
    #     else:
    #         hil = signals_band.get_data()[0, 0, :]
    #         base = data_padded[0, 0, :] - np.average(data_padded[0,0,:])
    #
    #     fig.add_trace(go.Scatter(y=np.abs(hil), name="env_band"+str(i), opacity=0.6))
    #     fig.add_trace(go.Scatter(y=np.real(hil), name="filt_band"+str(i), opacity=0.6))
    # fig.add_trace(go.Scatter(y=base, name="raw", opacity=0.6))
    #
    # fig.show("browser")


    '''
    # Removes the padding.
    if padding and padding > 0:
        tmin    = data.times [  padding ]
        tmax    = data.times [ -padding ]
        data    = data.crop ( tmin = tmin, tmax = tmax, include_tmax = False )
    '''

    # Gets the metadata.
    nsamp = data.get_data().shape[-1]
    nnode = data.get_data().shape[-2]
    nbands = len(bands)

    # Preallocate super-matrix
    cross_aec = np.zeros((nnode*nbands, nnode*nbands))

    for b1, _ in enumerate(bands):

        for b2, _ in enumerate(bands):

            if (b2 == b1):  # If intraband: Compute orth. cAEC

                # Extracts the raw data.
                rawdata = data_filt[b1].get_data()

                # Defines the minimum possible value.
                tiny = np.finfo(rawdata.dtype).eps

                # Defines the padding and the smoothing in samples.
                spadd = padding

                # Rewrites as channels x repetitions x samples.
                rawdata = rawdata.reshape([-1, nnode, nsamp])
                rawdata = np.swapaxes(rawdata, 0, 1)
                nrep = rawdata.shape[1]

                # Initializes the AEC matrices.
                aecov = np.zeros((nnode, nnode, nrep))
                aecovlc = np.zeros((nnode, nnode, nrep))
                nreg = np.zeros((nnode, nnode, nrep))

                # Splits in real and imaginary parts.
                rawreal = np.real(rawdata)
                rawimag = np.imag(rawdata)

                # Gets the envelope of the signal.
                rawenv = np.abs(rawdata)

                # Smooths the envelope and removes the padding.
                # rawenv = auxaec.get_mas(rawenv, ssmooth, spadd)

                # Removes the padding.
                rawenv  = rawenv [..., padding : -padding ]

                # Centers the envelope (as continuous or per repetitions).
                if continuous:
                    rawenv  = rawenv - rawenv.mean ( axis = -1, keepdims = True ).mean ( axis = -2, keepdims = True )
                    # rawenv = auxaec.demean2(rawenv)
                else:
                    rawenv  = rawenv - rawenv.mean ( axis = -1, keepdims = True )
                    # rawenv = auxaec.demean(rawenv)

                # Gets the AE covariance by matrix multiplication.
                for irep in range(nrep):
                    aecov[:, :, irep] = np.inner(rawenv[:, irep, :], rawenv[:, irep, :])

                # Gets the norm per node and repetition.
                dummy   = rawenv * rawenv
                nraw    = dummy.sum ( axis = -1 )
                # nraw = auxaec.get_norm(rawenv)

                # Concatenates all the repetitions to estimate the projections.
                rawconc = rawreal[..., spadd: -spadd or None].reshape([nnode, -1])
                rawconc = rawconc - rawconc.mean(axis=-1, keepdims=True)

                # Calculates the betas of the regression.
                # proj    = np.inner ( rawconc, rawconc )
                proj = rawconc.dot(rawconc.T)
                betas = proj / np.diag(proj)

                # Goes through each signal.
                for inode in range(nnode):

                    # Removes the projection of the current signal from all the others.
                    regreal = rawreal - betas[:, [inode], None] * rawreal[[inode], :, :]
                    regimag = rawimag - betas[:, [inode], None] * rawimag[[inode], :, :]
                    # regreal = auxaec.get_lc ( rawreal, rawreal [ inode, :, : ], betas [ :, inode ] )
                    # regimag = auxaec.get_lc ( rawimag, rawimag [ inode, :, : ], betas [ :, inode ] )

                    # Gets the envelope.
                    # regenv = auxaec.get_abs(regreal, regimag)
                    regenv = np.sqrt(regreal ** 2 + regimag ** 2)

                    # Smooths the envelope and removes the padding.
                    # regenv = auxaec.get_mas(regenv, ssmooth, spadd)

                    # Removes the padding.
                    regenv  = regenv [..., spadd : -spadd ]

                    # Centers the envelope (as continuous or per repetitions).
                    if continuous:
                        regenv  = regenv - regenv.mean ( axis = -1, keepdims = True ).mean ( axis = -2, keepdims = True )
                        # regenv = auxaec.demean2(regenv)
                    else:
                        regenv  = regenv - regenv.mean ( axis = -1, keepdims = True )
                        # regenv = auxaec.demean(regenv)

                    # Gets the AECov per repetition by matrix multiplication.
                    dummy   = rawenv [ [ inode ], :, : ] * regenv
                    aecovlc [ inode, :, : ] = dummy.sum ( axis = -1 )
                    # aecovlc[inode, :, :] = auxaec.get_dot(regenv, rawenv[inode, :, :])

                    # Gets the norm per node and repetition.
                    dummy   = regenv * regenv
                    nreg [ inode, :, : ] = dummy.sum ( axis = -1 )
                    # nreg[inode, :, :] = auxaec.get_norm(regenv)

                # If requested, simulates continuous data.
                if continuous:

                    # Gets the covariance and norms of the continuous data.
                    aecov = aecov.sum(axis=-1)
                    aecovlc = aecovlc.sum(axis=-1)
                    nraw = nraw.sum(axis=-1)
                    nreg = nreg.sum(axis=-1)

                    # Calculates the AEC of the continuous data as the normalized covariance.
                    aec = aecov / (tiny * tiny + np.sqrt(nraw[:, None] * nraw))
                    aeclc = aecovlc / (tiny * tiny + np.sqrt(nraw[:, None] * nreg))

                    # Makes the leakage-corrected AEC symmetric.
                    aeclc = (aeclc + aeclc.T) / 2

                    # Sets the number of epochs to 1.
                    nrep = 1

                # Otherwise treats each trial individually.
                else:

                    # Calculates the AEC as the normalized covariance.
                    aec = aecov / (tiny * tiny + np.sqrt(nraw[:, None, :] * nraw[None, :, :]))
                    aeclc = aecovlc / (tiny * tiny + np.sqrt(nraw[:, None, :] * nreg))

                    # Reshapes the AEC matrices as repetitions by nodes.
                    aec = np.swapaxes(aec, -1, 0)
                    aeclc = np.swapaxes(aeclc, -1, 0)

                    # Makes the leakage-corrected AEC symmetric.
                    aeclc = (aeclc + np.swapaxes(aeclc, -2, -1)) / 2

                # Forces the diagonal of the leakage-corrected AEC to be 0.
                diag = np.diag_indices(nnode)
                aeclc[..., diag[0], diag[1]] = 0

                # Flattens the connectivity matrices.
                # aec = aec.reshape((nrep, nnode * nnode))
                # aeclc = aeclc.reshape((nrep, nnode * nnode))
                aeclc = np.average(aeclc, axis=0)

                cross_aec[b1*nnode:b1*nnode+nnode, b2*nnode:b2*nnode+nnode] = aeclc

            elif (b2 > b1):  # Compute CrossFrequency

                # Extracts raw data and gets envelops
                rawdata1 = data_filt[b1].get_data()
                # Rewrites as channels x repetitions x samples.
                rawdata1 = rawdata1.reshape([-1, nnode, nsamp])
                rawdata1 = np.swapaxes(rawdata1, 0, 1)

                rawdata2 = data_filt[b2].get_data()
                rawdata2 = rawdata2.reshape([-1, nnode, nsamp])
                rawdata2 = np.swapaxes(rawdata2, 0, 1)

                nrep = rawdata1.shape[1]

                # Gets the envelope of the signal. and Removes the padding.
                raw1env = np.abs(rawdata1)[..., padding : -padding ]
                raw2env = np.abs(rawdata2)[..., padding : -padding ]

                # Centers the envelope (as continuous or per repetitions).
                if continuous:
                    raw1env = raw1env - raw1env.mean(axis=-1, keepdims=True).mean(axis=-2, keepdims=True)
                    raw2env = raw2env - raw2env.mean(axis=-1, keepdims=True).mean(axis=-2, keepdims=True)
                    # rawenv = auxaec.demean2(rawenv)
                else:
                    raw1env = raw1env - raw1env.mean(axis=-1, keepdims=True)
                    raw2env = raw2env - raw2env.mean(axis=-1, keepdims=True)
                    # rawenv = auxaec.demean(rawenv)

                aec = np.array([[[np.corrcoef(env1, env2)[0, 1] for env2 in raw2env[:, r, :]]
                                 for env1 in raw1env[:, r, :]] for r in range(nrep)])

                # Average our trials
                aec = np.average(aec, axis=0)

                cross_aec[b1*nnode:b1*nnode+nnode, b2*nnode:b2*nnode+nnode] = aec

    # Returns the estimated connectivity.
    return cross_aec





