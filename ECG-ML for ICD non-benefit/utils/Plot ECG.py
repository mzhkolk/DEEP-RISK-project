import matplotlib.pyplot as plt
import numpy as np
import pydicom
import h5py

LEAD_ORDER = ['Lead I (Einthoven)', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF', 'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6']

def plot_ecg(data, **kwargs):
    """Plots an ECG from either a numpy array or pydicom (DICOM) object."""
    if type(data) == pydicom.dataset.FileDataset:
        return plot_ecg_dicom(data, **kwargs)
    elif type(data) == np.ndarray:
        return plot_ecg_array(data, **kwargs)

def plot_ecg_array(data, block=True, seriesInstanceUID=None, patientID=None, label=None, pred=None):
    """
    Plot ECG from numpy array.

    Args:
        data (numpy.ndarray): Numpy array of size (samples x leads).
        block (bool): Pause python while plot is open if True.
        seriesInstanceUID (str): ECG identifier.
        patientID (str): Patient ID for the ECG.
        label (int): True label for ECG.
        pred (int): Predicted label for ECG.

    Returns:
        matplotlib.pyplot.Figure: Figure object for the plot.
    """
    N_samples = data.shape[0]
    fig, axs = plt.subplots(6, 2, figsize=(18,13))
    title = ''
    title += f'Series UID: {seriesInstanceUID}' if seriesInstanceUID is not None else ''
    title += f'\nPatientID: {patientID}' if patientID is not None else ''
    title += f'\nLabel: {str(label)}' if label is not None else ''
    title += f'\nPrediction: {str(pred)}' if pred is not None else ''
    fig.suptitle(title)
    for i, ax in enumerate(axs.T.flatten()):
        # Get Lead name
        source = LEAD_ORDER[i].split(' ')[1]
        # Plot lead
        ax.plot(data[:,i], linewidth=1)
        # Configure y-axis
        ax.set_ylabel(f'$\\bf{source}$', rotation=0)
        ax.set_ylim([-1500,1500])
        ax.yaxis.set_ticks(np.arange(-1500, 1500, 300))
        ax.yaxis.set_ticklabels([])
        # Configure x-axis
        ax.set_xlim([0,N_samples])
        ax.xaxis.set_ticks(np.arange(0, N_samples, 100))
        ax.xaxis.set_ticklabels([])
        # Other
        ax.grid()
        ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.show(block=block)
    return fig

def plot_ecg_dicom(dcm, wf_index=0, block=True):
    """
    Plot ECG from DICOM.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM file.
        wf_index (int): ECG waveform index. Usually 0 for raw ECG, 1 for median beat.
        block (bool): Pause python while plot is open if True.

    Returns:
        matplotlib.pyplot.Figure: Figure object for the plot.
    """
    multiplex = dcm.WaveformSequence[wf_index]
    N_samples = multiplex.NumberOfWaveformSamples
    arr = dcm.waveform_array(wf_index)
    fig, axs = plt.subplots(6, 2, figsize=(18,13))
    fig.suptitle(f'Series UID: {dcm.SeriesInstanceUID}\nPatientID: {dcm.PatientID}')
    for i, ax in enumerate(axs.T.flatten()):
        # Get Lead name
        source = multiplex.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning
        source = source.split(' ')[1]
        # Plot lead
        ax.plot(arr[:,i], linewidth=1)
        # Configure y-axis
        ax.set_ylabel(f'$\\bf{source}$', rotation=0)
        ax.set_ylim([-1500,1500])
        ax.yaxis.set_ticks(np.arange(-1500, 1500, 300))
        ax.yaxis.set_ticklabels([])
        # Configure x-axis
        ax.set_xlim([0,N_samples])
        ax.xaxis.set_ticks(np.arange(0, N_samples, 100))
        ax.xaxis.set_ticklabels([])
        # Other
        ax.grid()
        ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.show(block=block)
    return fig

def cycle_ecgs(dcm_gen, wf_index=0):
    """
    Cycle through ECG plots using a generator of DICOMs by pressing any button to go to next ECG.
    NOTE: THIS WORKS BUT WILL CRASH UPON TRYING TO QUIT!
    """
    for i, (dcm, _) in enumerate(dcm_gen):
        print(f'{i}:\tPatientID {dcm.PatientID}\tSeries UID {dcm.SeriesInstanceUID}')
        fig = plot_ecg(dcm, wf_index=wf_index, block=False)
        # Wait for keyboard button press (not mouse click) to close
        while True:
            if plt.waitforbuttonpress(0) == True:
                plt.close()
                break

def cycle_ecgs_hdf5(path, split='test', seriesInstanceUIDs=None, patientIDs=None, preds=None, start_idx=0):
    """
    Cycle through ECG plots using ECGs in HDF5 file.
    NOTE: THIS WORKS BUT WILL CRASH UPON TRYING TO QUIT!

    Args:
        path (str): Path to HDF5 file containing ECGs.
        split (str): Name of the split group name such as 'train', 'val' or 'test'.
        seriesInstanceUID (str): ECG identifier.
        patientID (str): Patient ID for the ECG.
        pred (int): Predicted label for ECG.
        start_idx (int): ECG index to start from.

    Returns:
        matplotlib.pyplot.Figure: Figure object for the plot.
    """
    with h5py.File(path, 'r') as f:
        data = f[f'{split}/data']
        labels = f[f'{split}/labels']
        N = len(data)
        for i in range(start_idx, N):
            ecg = data[i, :]
            seriesInstanceUID = seriesInstanceUIDs[i] if seriesInstanceUIDs is not None else None
            patientID = patientIDs[i] if patientIDs is not None else None
            prediction = preds[i] if preds is not None else None
            print(f'{i}:\tPatientID {patientID}\tSeries UID {seriesInstanceUID}')
            fig = plot_ecg(ecg, seriesInstanceUID=seriesInstanceUID, patientID=patientID, label=labels[i], block=False, pred=prediction)
            # Wait for keyboard button press (not mouse click) to close
            while True:
                if plt.waitforbuttonpress(0) == True:
                    plt.close()
                    break

if __name__ == '__main__':

    hdf5_path = r'Path\to\data.hdf5'
    cycle_ecgs_hdf5(hdf5_path)
