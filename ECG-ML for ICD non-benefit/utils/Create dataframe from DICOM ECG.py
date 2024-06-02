from pathlib import Path
from pydicom import dcmread
import glob
import pandas as pd

# Which data attributes to read from the dicom files.
DATA_ATTRIBUTES = ['SeriesInstanceUID', 'StudyDate', 'StudyTime', 'Modality', 'PatientID', 'PatientBirthDate', 'PatientSex', 'PatientAge', 'PatientSize', 'PatientWeight']

def folder_dcm_filenames(dir_path, recursive=True):
    """
    Creates generator with Path objects for each DICOM file in the specified directory.

    Args:
        dir_path (str, Path): Path to DICOM data directory.
        recursive (bool): If True, look through directory recursively.

    Returns:
        Generator with DICOM file Path objects.
    """
    assert type(dir_path) in [str, Path], 'Path to directory must be string or Path object'
    dir_path = Path(dir_path)
    if recursive:
        dcm_files = dir_path.rglob('*.dcm')
    else:
        dcm_files = dir_path.glob('*.dcm')
    return dcm_files

def read_dcm_folder(dir_path, recursive=True):
    """
    Creates a generator with parsed DICOM files.

    Args:
        dir_path (str, Path): Path to DICOM data directory.
        recursive (bool): If True, look through directory recursively.

    Yields:
        List of pydicom.dataset.FileDataset objects.
    """
    assert type(dir_path) in [str, Path], 'Path to directory must be string or Path object'
    dcm_files = folder_dcm_filenames(dir_path, recursive=recursive)
    for fpath in dcm_files:
        yield dcmread(fpath), fpath

def dcm_ecg_annotations(dcm):
    """
    Read annotations in DICOM, ignoring Pacemaker Spike annotations.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM file.

    Returns:
        Dictionary: Dictionary with (annotation_name, annotation_value) key-value pairs.
    """
    # Check if annotations are present
    try:
        annot_sequence = dcm['WaveformAnnotationSequence']
    except:
        return None
    annotations = dict()
    # For each annotation, try if there are basic automatic annotations (e.g. heart rate, PR Interval, T Axis)
    for annot in annot_sequence:
        try:
            name = annot.ConceptNameCodeSequence[0].CodeMeaning
            if name == 'Pacemaker Spike': continue # Ignore pacemaker annotations for now
            value = float(annot.get('NumericValue'))
        except:
            continue
        annotations[name] = value
    return annotations

def dcm_to_dict(dcm):
    """
    Return DICOM attributes as a dictionary.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM file.

    Returns:
        Dictionary: Dictionary with (attribute_name, attribute_value) key-value pairs.
    """
    data = {attr: dcm.get(attr, default=None) for attr in DATA_ATTRIBUTES}
    data['NumECGs'] = 1
    waveform = dcm.WaveformSequence[0]
    data['SampleDuration'] = waveform.NumberOfWaveformSamples
    data['SamplingFrequency'] = waveform.SamplingFrequency
    data['NumChannels'] = waveform.NumberOfWaveformChannels
    # Update data dictionary with ecg annotations
    annotations = dcm_ecg_annotations(dcm)
    if annotations is not None: data = {**data, **annotations}
    return data

def read_dcm_file(path):
    """Read/parse DICOM file using pydicom libary."""
    dcm = dcmread(path)
    return dcm

def df_from_dcm(dcm_generator, use='all'):
    """
    Create pandas DataFrame from DICOM files.

    Args:
        dcm_generator (generator): Generator for (pydicom.dataset.FileDataset, file_path) tuples.
        use (str): One of ['all', 'first', 'latest'], indicates which dicom files to include
            per patient, all, only first, or chronologically last study. Default to 'all'.

    Returns:
        pandas.Dataframe object containing dicom file information.
    """
    assert use in ['all', 'first', 'latest'], 'Argument \'use\' must be one of [all, first, latest].'

    datadict = dict() # Dictionary seriesID or patientID : dictionary with dcm data
    patient2date = dict() # Dictionary patientID : (StudyDate, list(seriesIDs))

    # Enumerate over DICOM files
    for i, (dcm, path) in enumerate(dcm_generator):
        # Acquire data
        seriesID = dcm.SeriesInstanceUID
        patientID = dcm.PatientID
        date = dcm.StudyDate
        data = dcm_to_dict(dcm)
        data['FileName'] = path

        # Use all patient ECGs, dictionary becomes seriesID oriented
        if use == 'all':
            assert seriesID not in datadict, f'Duplicate seriesID {seriesID}'
            datadict[seriesID] = data
        # For each patient, use only the first or latest ECG, dictionary patientID oriented
        else:
            # If patientID not seen before, use this ECG
            if patientID not in patient2date:
                patient2date[patientID] = (date, [seriesID])
                datadict[patientID] = data
            # If patientID already seen before, use either latest or first ECG
            else:
                prev_date, seriesIDs = patient2date[patientID]
                assert seriesID not in seriesIDs, f'Duplicate seriesID {seriesID}'
                seriesIDs.append(seriesID)
                if (use == 'first' and date < prev_date) or (use == 'latest' and date > prev_date):
                    patient2date[patientID] = (date, seriesIDs)
                    datadict[patientID] = data
                else:
                    patient2date[patientID] = (prev_date, seriesIDs)
                datadict[patientID]['NumECGs'] = len(seriesIDs)
    # Create Dataframe from the dictionaries
    df = pd.DataFrame.from_dict(datadict, orient='index')
    # Reset index to integers, patientID and seriesID already in dataframe
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':

    data_path = r'Path\to\directory\with\dicoms'
    dcm_gen = read_dcm_folder(data_path, recursive=True)

    df = df_from_dcm(dcm_gen, use='all')
    print(len(df))
    print(df.head())
