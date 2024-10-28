from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from obspy import Stream,Trace

from .base import BenchmarkDataset, WaveformDataWriter

class DiTing(BenchmarkDataset):
    """
        A chunked dummy dataset visualizing the implementation of custom datasets with chunking
        """

    def __init__(self, **kwargs):
        citation = (
            "Zhao M, Xiao Z W, Chen S and Fang L H (2022). Diting: a large-scale chinese seismic benchmark  "
            "dataset  for  artificial intelligence in seismology. Earthq Sci 35, doi:10.12080/nedc.11.ds.2022.0002"
        )


        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        sampling_rate = 100

        def _get_from_DiTing(part = None,
                            key = None,
                            h5file_path = basepath,):

            with h5py.File(h5file_path + '/waveforms_{}.hdf5'.format(part), 'r') as f:

                dataset = f.get('earthquake/' + str(key))
                data = np.array(dataset).astype(np.float32).T

                return data

        def _get_from_DiTing_noise(part = None,
                            key = None,
                            h5file_path = basepath,):

            with h5py.File(h5file_path + '/noise_{}.hdf5'.format(part), 'r') as f:

                dataset = f.get('noise/' + str(key))
                data = np.array(dataset).astype(np.float32).T

                return data

        metadata_dict = {
            "key": "key",
            "ev_id": "source_id",
            "evmag": "source_event_magnitude",
            "mag_type": "source_magnitude_type",
            "p_pick": "trace_p_arrival_sample",
            "p_clarity": "trace_p_clarity",
            "p_motion": "trace_p_motion",
            "s_pick": "trace_s_arrival_sample",
            "net": "station_network_code",
            "sta_id": "station_id",
            "dis": "source_distance",
            "st_mag": "station_magnitude",
            "baz": "baz",
            "Z_P_amplitude_snr": "trace_Z_P_amplitude_snr",
            "Z_P_power_snr": "trace_Z_P_power_snr",
            "Z_S_amplitude_snr": "trace_Z_S_amplitude_snr",
            "Z_S_power_snr": "trace_Z_S_power_snr",
            "N_P_amplitude_snr": "trace_N_P_amplitude_snr",
            "N_P_power_snr": "trace_N_P_power_snr",
            "N_S_amplitude_snr": "trace_N_S_amplitude_snr",
            "N_S_power_snr": "trace_N_S_power_snr",
            "E_P_amplitude_snr": "trace_E_P_amplitude_snr",
            "E_P_power_snr": "trace_E_P_power_snr",
            "E_S_amplitude_snr": "trace_E_S_amplitude_snr",
            "E_S_power_snr": "trace_E_S_power_snr",
            "P_residual": "trace_P_residual",
            "S_residual": "trace_S_residual"
        }

        if basepath is None:
            raise ValueError(
                "No cached version of DiTing found. "
            )

        basepath = Path(basepath)

        metadata = pd.read_csv(basepath / "DiTing330km_total.csv",low_memory=False)
        metadata.rename(columns=metadata_dict, inplace=True)


        # spilt dataset
        # Shuffle the metadata randomly
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

        # Set the proportion of data for each split
        train_ratio = 0.7  # 80% for training
        dev_ratio = 0.2  # 10% for development

        # Calculate the number of samples for each split
        num_samples = len(metadata)
        num_train = int(train_ratio * num_samples)
        num_dev = int(dev_ratio * num_samples)

        # Assign split labels to the metadata
        metadata.loc[:num_train, "split"] = "train"
        metadata.loc[num_train:num_train + num_dev, "split"] = "dev"
        metadata.loc[num_train + num_dev:, "split"] = "test"

        metadata = metadata.sort_values(by=metadata.columns[0], ascending=True)

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": sampling_rate,
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        writer.set_total(len(metadata))

        for _, row in metadata.iterrows():
            row = row.to_dict()

            trace_category  = row['trace_category']

            part = row['part']
            key = row["key"]
            key_correct = str(key).split('.')
            key = key_correct[0].rjust(6, '0') + '.' + key_correct[1].ljust(4, '0')

            if trace_category == 'earthquake_local':
                waveforms = _get_from_DiTing(part=part, key=key)
            elif trace_category == 'noise':
                waveforms = _get_from_DiTing_noise(part=part,key=key)

            writer.add_trace(row, waveforms)

class OriginalDiTing(BenchmarkDataset):
    """
        A chunked dummy dataset visualizing the implementation of custom datasets with chunking
        """

    def __init__(self, **kwargs):
        citation = (
            "Zhao M, Xiao Z W, Chen S and Fang L H (2022). Diting: a large-scale chinese seismic benchmark  "
            "dataset  for  artificial intelligence in seismology. Earthq Sci 35, doi:10.12080/nedc.11.ds.2022.0002"
        )


        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        sampling_rate = 50

        def _get_from_DiTing(part = None,
                            key = None,
                            h5file_path = basepath,):

            with h5py.File(h5file_path + '/waveforms_{}.hdf5'.format(part), 'r') as f:
                dataset = f.get('earthquake/' + str(key))
                data = np.array(dataset).astype(np.float32).T
                return data



        metadata_dict = {
            "key": "key",
            "ev_id": "source_id",
            "evmag": "source_event_magnitude",
            "mag_type": "source_magnitude_type",
            "p_pick": "trace_p_arrival_sample",
            "p_clarity": "trace_p_clarity",
            "p_motion": "trace_p_motion",
            "s_pick": "trace_s_arrival_sample",
            "net": "station_network_code",
            "sta_id": "station_id",
            "dis": "source_distance",
            "st_mag": "station_magnitude",
            "baz": "baz",
            "Z_P_amplitude_snr": "trace_Z_P_amplitude_snr",
            "Z_P_power_snr": "trace_Z_P_power_snr",
            "Z_S_amplitude_snr": "trace_Z_S_amplitude_snr",
            "Z_S_power_snr": "trace_Z_S_power_snr",
            "N_P_amplitude_snr": "trace_N_P_amplitude_snr",
            "N_P_power_snr": "trace_N_P_power_snr",
            "N_S_amplitude_snr": "trace_N_S_amplitude_snr",
            "N_S_power_snr": "trace_N_S_power_snr",
            "E_P_amplitude_snr": "trace_E_P_amplitude_snr",
            "E_P_power_snr": "trace_E_P_power_snr",
            "E_S_amplitude_snr": "trace_E_S_amplitude_snr",
            "E_S_power_snr": "trace_E_S_power_snr",
            "P_residual": "trace_P_residual",
            "S_residual": "trace_S_residual"
        }

        if basepath is None:
            raise ValueError(
                "No cached version of DiTing found. "
            )

        basepath = Path(basepath)

        metadata = pd.read_csv(basepath / "DiTing330km_total.csv",low_memory=False)
        metadata.rename(columns=metadata_dict, inplace=True)

        # spilt dataset
        # Shuffle the metadata randomly
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

        # Set the proportion of data for each split
        train_ratio = 0.7  # 80% for training
        dev_ratio = 0.2  # 10% for development

        # Calculate the number of samples for each split
        num_samples = len(metadata)
        num_train = int(train_ratio * num_samples)
        num_dev = int(dev_ratio * num_samples)

        # Assign split labels to the metadata
        metadata.loc[:num_train, "split"] = "train"
        metadata.loc[num_train:num_train + num_dev, "split"] = "dev"
        metadata.loc[num_train + num_dev:, "split"] = "test"

        metadata = metadata.sort_values(by=metadata.columns[0], ascending=True)
        metadata['trace_p_arrival_sample'] *= 2
        metadata['trace_s_arrival_sample'] *= 2


        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": sampling_rate,
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        writer.set_total(len(metadata))

        for _, row in metadata.iterrows():
            row = row.to_dict()

            part = row['part']
            key = row["key"]
            key_correct = str(key).split('.')
            key = key_correct[0].rjust(6, '0') + '.' + key_correct[1].ljust(4, '0')

            waveforms = _get_from_DiTing(part=part, key=key)

            writer.add_trace(row, waveforms)
