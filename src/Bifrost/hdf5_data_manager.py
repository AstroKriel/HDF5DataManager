import copy
import unittest
import numpy as np
from enum import Enum


class AxisUnits(Enum):
    NOT_SPECIFIED = "not_specified"
    DIMENSIONLESS = "dimensionless"
    T_TURB = "t_turb"
    K_TURB = "k_turb"

class DatasetUnits(Enum):
    NOT_SPECIFIED = "not_specified"
    DIMENSIONLESS = "dimensionless"


class AxisObject:
    def __init__(self, group, name, values, units=AxisUnits.NOT_SPECIFIED, notes=""):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        self.units  = units
        self.locked = False
        self.notes  = notes

    def add(self, values_in, bool_overwrite=False):
        if self.locked: raise ValueError(f"Axis `/axes/{self.group}/{self.name}` is locked and cannot be modified.")
        self.values = np.unique(np.concatenate((self.values, np.array(values_in))))


class DatasetObject:
    def __init__(self, group, name, values, list_axis_objs, units=DatasetUnits.NOT_SPECIFIED, notes=""):
        self.group  = group
        self.name   = name
        self.data   = np.array(values)
        self.units  = units
        self.notes  = notes
        self.locked = False
        self.list_axis_objs_copy = copy.deepcopy(list_axis_objs)

    def add(self, dataset_values_in, list_axis_values_in, list_axis_objs_updated, bool_overwrite=False):
        if self.locked: raise ValueError(f"Dataset `/datasets/{self.group}/{self.name}` is locked and cannot be modified.")
        self.reindex(list_axis_values_in, list_axis_objs_updated, dataset_values_in)

    def reindex(self, list_axis_values_in, list_axis_objs_updated, dataset_values_in=None):
        list_axis_values_old = [
            np.array(obj_axis_old.values, copy=True)
            for obj_axis_old in self.list_axis_objs_copy
        ]
        self.list_axis_objs_copy = copy.deepcopy(list_axis_objs_updated)
        updated_dataset_shape = tuple(
            len(obj_axis_updated.values)
            for obj_axis_updated in self.list_axis_objs_copy
        )
        dataset_values_updated = np.full(updated_dataset_shape, np.nan)
        dataset_indices_old = tuple(
            np.searchsorted(obj_axis_updated.values, old_axis_values)
            for obj_axis_updated, old_axis_values in zip(self.list_axis_objs_copy, list_axis_values_old)
        )
        dataset_values_updated[np.ix_(*dataset_indices_old)] = self.data
        if dataset_values_in is not None:
            dataset_indices_new = tuple(
                np.searchsorted(obj_axis_updated.values, new_axis_values)
                for obj_axis_updated, new_axis_values in zip(self.list_axis_objs_copy, list_axis_values_in)
            )
            dataset_values_updated[np.ix_(*dataset_indices_new)] = dataset_values_in
        self.data = dataset_values_updated


class HDF5DataManager:
    def __init__(self):
        self.dict_axes = {}  # {group: {name: AxisObject}}
        self.dict_datasets = {}  # {group: {name: DatasetObject}}
        self.dict_axis_dependencies = {}  # { (axis_group, axis_name): [(dataset_group, dataset_name), ...] }

    def add_data(self, dict_dataset, list_axis_dicts, bool_overwrite=False):
        ## there needs to be alignment between dataset and axis:
        ## 1. dict_dataset["values"].ndim == len(list_axis_dicts)
        ## 2. dict_dataset["values"].shape[i] == len(list_axis_dicts[i]["values"])
        ## the following properties should be gauranteed, at least with default values defined by the class that creates the dict
        dataset_group  = dict_dataset.get("group")
        dataset_name   = dict_dataset.get("name")
        dataset_values = dict_dataset.get("values")
        dataset_units  = dict_dataset.get("units")
        list_axis_objs = [
            self._create_or_update_axis(dict_axis, bool_overwrite)
            for dict_axis in list_axis_dicts
        ]
        for obj_axis in list_axis_objs:
            axis_id    = (obj_axis.group, obj_axis.name)
            dataset_id = (dataset_group, dataset_name)
            if axis_id not in self.dict_axis_dependencies:
                self.dict_axis_dependencies[axis_id] = []
            if dataset_id not in self.dict_axis_dependencies[axis_id]:
                self.dict_axis_dependencies[axis_id].append(dataset_id)
        if dataset_group not in self.dict_datasets:
            self.dict_datasets[dataset_group] = {}
        if dataset_name not in self.dict_datasets[dataset_group]:
            ## create dataset for the first time
            self.dict_datasets[dataset_group][dataset_name] = DatasetObject(
                group  = dataset_group,
                name   = dataset_name,
                values = dataset_values,
                units  = dataset_units,
                list_axis_objs = list_axis_objs
            )
        else:
            ## merge new data into the existing dataset
            list_axis_values = [
                dict_axis.get("values")
                for dict_axis in list_axis_dicts
            ]
            self.dict_datasets[dataset_group][dataset_name].add(
                dataset_values_in      = dataset_values,
                list_axis_values_in    = list_axis_values, # input axes values that have not been merged with existing axes
                list_axis_objs_updated = list_axis_objs,
                bool_overwrite         = bool_overwrite
            )
        self._check_axis_dependency_and_reindex_where_necessary(list_axis_objs)

    def _create_or_update_axis(self, dict_axis, bool_overwrite=False):
        ## the following properties should be gauranteed, at least with default values defined by the class that creates the dict
        axis_group  = dict_axis.get("group")
        axis_name   = dict_axis.get("name")
        axis_values = dict_axis.get("values")
        axis_units  = dict_axis.get("units")
        if axis_group not in self.dict_axes:
            self.dict_axes[axis_group] = {}
        if axis_name not in self.dict_axes[axis_group]:
            self.dict_axes[axis_group][axis_name] = AxisObject(
                group  = axis_group,
                name   = axis_name,
                values = axis_values,
                units  = axis_units
            )
        else: self.dict_axes[axis_group][axis_name].add(axis_values, bool_overwrite)
        return self.dict_axes[axis_group][axis_name]

    def _check_axis_dependency_and_reindex_where_necessary(self, updated_axes):
        for axis in updated_axes:
            axis_key = (axis.group, axis.name)
            if axis_key in self.dict_axis_dependencies:
                for dataset_group, dataset_name in self.dict_axis_dependencies[axis_key]:
                    dataset = self.dict_datasets[dataset_group][dataset_name]
                    new_axis_values = [
                        self.dict_axes[axis.group][axis.name].values
                        for axis in dataset.list_axis_objs_copy
                    ]
                    dataset.reindex(
                        list_axis_values_in    = new_axis_values,
                        list_axis_objs_updated = dataset.list_axis_objs_copy,
                        dataset_values_in      = dataset.data
                    )

    def lock_axis(self, axis_group, axis_name):
        if axis_group in self.dict_axes and axis_name in self.dict_axes[axis_group]:
            self.dict_axes[axis_group][axis_name].locked = True
        else: raise ValueError(f"Axis `/axes/{axis_group}/{axis_name}` does not exist.")

    def lock_dataset(self, dataset_group, dataset_name):
        if dataset_group in self.dict_datasets and dataset_name in self.dict_datasets[dataset_group]:
            self.dict_datasets[dataset_group][dataset_name].locked = True
        else: raise ValueError(f"Dataset `/datasets/{dataset_group}/{dataset_name}` does not exist.")

    def get_dataset(self, dataset_group, dataset_name):
        return self.dict_datasets[dataset_group][dataset_name].data

    def get_axis(self, axis_group, axis_name):
        return self.dict_axes[axis_group][axis_name].values

    @staticmethod
    def create_dict_dataset(group, name, values, units=DatasetUnits.NOT_SPECIFIED, notes=""):
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Dataset group must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if not isinstance(values, (list, np.ndarray)):
            raise TypeError("Dataset values must be a list or numpy array.")
        values = np.array(values)
        if not isinstance(units, DatasetUnits):
            raise TypeError(f"Invalid dataset unit: {units}. Must be an instance of DatasetUnits.")
        if not isinstance(notes, str):
            raise TypeError("Notes must be a string.")
        return {
            "group"  : group,
            "name"   : name,
            "values" : np.array(values),
            "units"  : units,
            "notes"  : notes
        }

    @staticmethod
    def create_dict_axis(group, name, values, units=AxisUnits.NOT_SPECIFIED, notes=""):
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Axis group must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Axis name must be a non-empty string.")
        if not isinstance(values, (list, np.ndarray)):
            raise TypeError("Axis values must be a list or numpy array.")
        values = np.array(values)
        if values.ndim != 1:
            raise ValueError("Axis values must be a 1D array.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Axis values must be numeric (either integers or floats).")
        if not isinstance(units, AxisUnits):
            raise TypeError(f"Invalid axis unit: {units}. Must be an instance of AxisUnits.")
        if not isinstance(notes, str):
            raise TypeError("Notes must be a string.")
        return {
            "group"  : group,
            "name"   : name,
            "values" : values,
            "units"  : units,
            "notes"  : notes
        }

class TestHDF5DataManager(unittest.TestCase):

    def setUp(self):
        self.h5dmanager = HDF5DataManager()

    # def test_create_dict_axis(self):
    #     axis_values = np.arange(1000)
    #     axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED, notes="Test axis")
    #     self.assertEqual(axis_dict["group"], "axis_group")
    #     self.assertEqual(axis_dict["name"], "axis_name")
    #     np.testing.assert_array_equal(axis_dict["values"], axis_values)
    #     self.assertEqual(axis_dict["units"], AxisUnits.NOT_SPECIFIED)
    #     self.assertEqual(axis_dict["notes"], "Test axis")

    # def test_create_dict_dataset(self):
    #     dataset_values = np.random.rand(1000) * 100
    #     dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.DIMENSIONLESS, notes="Test dataset")
    #     self.assertEqual(dataset_dict["group"], "dataset_group")
    #     self.assertEqual(dataset_dict["name"], "dataset_name")
    #     np.testing.assert_array_equal(dataset_dict["values"], dataset_values)
    #     self.assertEqual(dataset_dict["units"], DatasetUnits.DIMENSIONLESS)
    #     self.assertEqual(dataset_dict["notes"], "Test dataset")

    # def test_add_data(self):
    #     length = 100
    #     axis_values = np.arange(length)
    #     dataset_values = np.random.rand(length)
    #     axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
    #     dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
    #     self.h5dmanager.add_data(dataset_dict, [axis_dict], bool_overwrite=False)
    #     stored_data = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
    #     np.testing.assert_array_equal(stored_data, dataset_values)

    # def test_extend_data(self):
    #     length = 100
    #     axis_values1 = np.arange(length)
    #     dataset_values1 = np.random.rand(length)
    #     axis_dict1 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values1, AxisUnits.NOT_SPECIFIED)
    #     dataset_dict1 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values1, DatasetUnits.NOT_SPECIFIED)
    #     self.h5dmanager.add_data(dataset_dict1, [axis_dict1], bool_overwrite=False)
    #     axis_values2 = length + np.arange(length)
    #     dataset_values2 = np.random.rand(length)
    #     axis_dict2 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values2, AxisUnits.NOT_SPECIFIED)
    #     dataset_dict2 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values2, DatasetUnits.NOT_SPECIFIED)
    #     self.h5dmanager.add_data(dataset_dict2, [axis_dict2], bool_overwrite=False)
    #     stored_data = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
    #     np.testing.assert_array_equal(stored_data, np.concatenate([dataset_values1, dataset_values2]))

    def test_reindexing_shared_axis(self):
        length = 100
        axis_values1 = np.arange(length)
        dataset_values1 = np.random.rand(length)
        axis_dict1 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values1, AxisUnits.NOT_SPECIFIED)
        dataset_dict1 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values1, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add_data(dataset_dict1, [axis_dict1], bool_overwrite=False)
        axis_values2 = length + np.arange(length)
        dataset_values2 = np.random.rand(length)
        axis_dict2 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values2, AxisUnits.NOT_SPECIFIED)
        dataset_dict2 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name2", dataset_values2, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add_data(dataset_dict2, [axis_dict2], bool_overwrite=False)
        stored_data = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(stored_data, np.concatenate([dataset_values1, dataset_values2]))

    # def test_add_2d_dataset(self):
    #     length_rows = 50
    #     length_cols = 100
    #     dict_axis_rows = HDF5DataManager.create_dict_axis("axis_group", "axis_rows", np.arange(length_rows), AxisUnits.NOT_SPECIFIED)
    #     dict_axis_cols = HDF5DataManager.create_dict_axis("axis_group", "axis_cols", np.arange(length_cols), AxisUnits.NOT_SPECIFIED)
    #     dataset_values_in = np.random.rand(length_rows, length_cols)
    #     dict_dataset = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values_in, DatasetUnits.NOT_SPECIFIED)
    #     self.h5dmanager.add_data(dict_dataset, [dict_axis_rows, dict_axis_cols], bool_overwrite=False)
    #     dataset_values_read = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
    #     np.testing.assert_array_equal(dataset_values_read, dataset_values_in)

if __name__ == '__main__':
    unittest.main()
