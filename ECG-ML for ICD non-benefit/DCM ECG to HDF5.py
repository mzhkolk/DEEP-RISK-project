import read_dcm
import h5py
import numpy as np
from tqdm import tqdm

# Fixed ordering of leads
LEAD_ORDER = ['Lead I (Einthoven)', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF', 'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6']

def create_hdf5(outpath, group_name, filenames, samples=5000, channels=12):
    """
    Creates/extends a HDF5 file and fills the corresponding group with ECGs
        from DICOM files.

    Args:
        outpath (str): Path for the HDF5 file.
        group_name (str): Group or dataset name in the HDF5 file to save ECGs to.
        filenames (list): List of paths to the DICOM files.
        samples (int): Number of recorded time points to read/store.
        channels (int): Number of channels\leads for the ECGs.
    """
    N = len(filenames)
    with h5py.File(outpath, 'a') as f:
        # Store ecg data
        dset_data = f.create_dataset(group_name, (N, samples, channels), dtype='f4')
        for i, fpath in tqdm(enumerate(filenames), total=len(filenames)):
            dcm = read_dcm.read_dcm_file(fpath)
            ecg = dcm.waveform_array(0)
            # Map each lead to an index in the array (to ensure uniform lead-order across all ecgs)
            lead2idx = {}
            for j, channel in enumerate(dcm.WaveformSequence[0].ChannelDefinitionSequence):
                source = channel.ChannelSourceSequence[0].CodeMeaning
                lead2idx[source] = j
            try:
                indices = [lead2idx[lead] for lead in LEAD_ORDER] # Try for "Lead I (Einthoven)", "Lead II" notation
            except KeyError:
                indices = [lead2idx[lead.split()[1]] for lead in LEAD_ORDER] # Try for "I", "II" notation
            dset_data[i, :, :] = ecg[:samples, indices]
