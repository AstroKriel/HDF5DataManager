from enum import Enum
import numpy as np

# Enums for Axis Types and Units
class AxisType(Enum):
    TIME = 'time'
    BIN_EDGES = 'bin_edges'
    WAVENUMBER = 'wavenumber'

class AxisUnits(Enum):
    SECONDS = 'seconds'
    TESLA = 'Tesla'
    KG_M3 = 'kg/m^3'
    ONE_OVER_M = '1/m'
    UNITLESS = 'unitless'

# DataManager Class
class DataManager:
    def __init__(self):
        self.axes = {}
        self.datasets = {}
        self.metadata = {}

    def add_axis(self, axis_type, name, units):
        if name in self.axes:
            raise ValueError(f"Axis '{name}' already exists.")
        self.axes[name] = {'type': axis_type, 'units': units}

    def add_dataset(self, dataset_type, name, data, axes_info):
        if len(axes_info) != len(data.shape):
            raise ValueError("Number of axes does not match data dimensions.")

        for axis in axes_info:
            axis_name = axis['name']
            if axis_name not in self.axes:
                self.add_axis(axis['type'], axis_name, axis.get('units', None))

        if dataset_type not in self.datasets:
            self.datasets[dataset_type] = {}
        self.datasets[dataset_type][name] = {
            'data': data,
            'axes': [axis['name'] for axis in axes_info]
        }

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def save_to_hdf5(self, file_path):
        # Placeholder for HDF5 save logic
        pass

    def load_from_hdf5(self, file_path):
        # Placeholder for HDF5 load logic
        pass

    def cull_unused_axes(self):
        used_axes = set()
        for dataset_type in self.datasets.values():
            for dataset in dataset_type.values():
                used_axes.update(dataset['axes'])

        for axis_name in list(self.axes.keys()):
            if axis_name not in used_axes:
                del self.axes[axis_name]

    def check_consistency(self):
        axis_lengths = {}
        for dataset_type in self.datasets.values():
            for dataset in dataset_type.values():
                for i, axis_name in enumerate(dataset['axes']):
                    if axis_name not in axis_lengths:
                        axis_lengths[axis_name] = len(dataset['data'].shape[i])
                    elif axis_lengths[axis_name] != len(dataset['data'].shape[i]):
                        raise ValueError(f"Inconsistent length for axis {axis_name}")
