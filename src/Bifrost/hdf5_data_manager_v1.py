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
        self.notes  = notes

    def add(self, values_in):
        ## the following is equivelant to np.array(sorted(set(self.values) | set(values_in)))
        self.values = np.unique(np.concatenate((self.values, np.array(values_in))))

    def __eq__(self, obj_axis):
        if isinstance(obj_axis, AxisObject):
            bool_same_properties = all([
                self.group == obj_axis.group,
                self.name == obj_axis.name,
                np.array_equal(self.values, obj_axis.values),
                self.units == obj_axis.units,
                self.notes == obj_axis.notes
            ])
            # print(f"axis `{self.group}/{self.name}` and `{obj_axis.group}/{obj_axis.name}` are {'the same' if bool_identical else 'different'}.") # for debugging
            return bool_same_properties
        return False

class DatasetObject:
    def __init__(self, group, name, values, list_axis_objs, units=DatasetUnits.NOT_SPECIFIED, notes=""):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        self.units  = units
        self.notes  = notes
        self.list_axis_objs_copy = copy.deepcopy(list_axis_objs)

    def add(self, dataset_values_in, list_axis_values_in, list_axis_objs_updated):
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
            for obj_axis_updated, old_axis_values in zip(
                self.list_axis_objs_copy,
                list_axis_values_old
            )
        )
        dataset_values_updated[np.ix_(*dataset_indices_old)] = self.values
        if dataset_values_in is not None:
            ## `in` may not necessarily be `new`. where `old` == `in` -> overwrite, and where `old` != `in` -> add/merge
            dataset_indices_in = tuple(
                np.searchsorted(obj_axis_updated.values, new_axis_values)
                for obj_axis_updated, new_axis_values in zip(
                    self.list_axis_objs_copy,
                    list_axis_values_in
                )
            )
            dataset_values_updated[np.ix_(*dataset_indices_in)] = dataset_values_in
        self.values = dataset_values_updated

class HDF5DataManager:
    def __init__(self):
        self.dict_axes = {}  # {group: {name: AxisObject}}
        self.dict_datasets = {}  # {group: {name: DatasetObject}}
        self.dict_axis_dependencies = {}  # { (axis_group, axis_name): [(dataset_group, dataset_name), ...] }

    def add(self, dict_dataset, list_axis_dicts):
        ## there needs to be alignment between dataset and axis:
        ## 1. dict_dataset["values"].ndim == len(list_axis_dicts)
        ## 2. dict_dataset["values"].shape[i] == len(list_axis_dicts[i]["values"])
        ## the following properties should be gauranteed, at least with default values defined by the class that creates the dict
        dataset_group  = dict_dataset.get("group")
        dataset_name   = dict_dataset.get("name")
        dataset_values = dict_dataset.get("values")
        dataset_units  = dict_dataset.get("units")
        list_axis_values_in = [
            dict_axis_in.get("values")
            for dict_axis_in in list_axis_dicts
        ]
        list_axis_objs_in = [
            self._create_axis(dict_axis_in)
            for dict_axis_in in list_axis_dicts
        ]
        list_axis_objs_updated = [
            self._create_or_update_axis(dict_axis_in)
            for dict_axis_in in list_axis_dicts
        ]
        for obj_axis in list_axis_objs_updated:
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
                list_axis_objs = list_axis_objs_in,
            )
        else:
            ## merge new values into the existing dataset
            self.dict_datasets[dataset_group][dataset_name].add(
                dataset_values_in      = dataset_values,
                list_axis_values_in    = list_axis_values_in, # input axes values that have not been merged with the existing axes
                list_axis_objs_updated = list_axis_objs_updated,
            )
        self._check_axis_dependency_and_reindex_where_necessary(
            list_axis_values_in    = list_axis_values_in,
            list_axis_objs_updated = list_axis_objs_updated,
        )

    def _create_axis(self, dict_axis):
        ## the following properties should be gauranteed, at least with default values defined by the class that creates the dict
        axis_group  = dict_axis.get("group")
        axis_name   = dict_axis.get("name")
        axis_values = dict_axis.get("values")
        axis_units  = dict_axis.get("units")
        return AxisObject(
            group  = axis_group,
            name   = axis_name,
            values = axis_values,
            units  = axis_units
        )

    def _create_or_update_axis(self, dict_axis):
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
        else: self.dict_axes[axis_group][axis_name].add(axis_values)
        return self.dict_axes[axis_group][axis_name]

    def _check_axis_dependency_and_reindex_where_necessary(self, list_axis_values_in, list_axis_objs_updated):
        for obj_axis_updated in list_axis_objs_updated:
            axis_id = (obj_axis_updated.group, obj_axis_updated.name)
            if axis_id in self.dict_axis_dependencies:
                for dataset_group, dataset_name in self.dict_axis_dependencies[axis_id]:
                    dataset = self.dict_datasets[dataset_group][dataset_name]
                    bool_axes_refs_up_to_date = dataset.list_axis_objs_copy == list_axis_objs_updated
                    bool_dataset_shape_matches_axes = dataset.values.shape == tuple(len(axis.values) for axis in list_axis_objs_updated)
                    if bool_axes_refs_up_to_date and bool_dataset_shape_matches_axes: continue
                    # print(f"reindexing: {dataset_group}/{dataset_name}") # for debugging
                    dataset.reindex(
                        list_axis_values_in    = list_axis_values_in,
                        list_axis_objs_updated = list_axis_objs_updated,
                    )

    def get_dataset(self, dataset_group, dataset_name):
        return self.dict_datasets[dataset_group][dataset_name].values

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

    def test_create_dict_axis(self):
        axis_values = np.arange(1000)
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED, notes="Test axis")
        self.assertEqual(axis_dict["group"], "axis_group")
        self.assertEqual(axis_dict["name"], "axis_name")
        np.testing.assert_array_equal(axis_dict["values"], axis_values)
        self.assertEqual(axis_dict["units"], AxisUnits.NOT_SPECIFIED)
        self.assertEqual(axis_dict["notes"], "Test axis")

    def test_create_dict_dataset(self):
        dataset_values = np.random.rand(1000) * 100
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.DIMENSIONLESS, notes="Test dataset")
        self.assertEqual(dataset_dict["group"], "dataset_group")
        self.assertEqual(dataset_dict["name"], "dataset_name")
        np.testing.assert_array_equal(dataset_dict["values"], dataset_values)
        self.assertEqual(dataset_dict["units"], DatasetUnits.DIMENSIONLESS)
        self.assertEqual(dataset_dict["notes"], "Test dataset")

    def test_add(self):
        length = 100
        axis_values = np.arange(length)
        dataset_values = np.random.rand(length)
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict, [axis_dict])
        stored_data = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(stored_data, dataset_values)

    def test_extend_data(self):
        length = 10
        axis_split_from_factor = 2
        index_start_extension_from = length // axis_split_from_factor
        axis_values1 = np.arange(length)
        dataset_values1 = np.ones(length)
        axis_dict1 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values1, AxisUnits.NOT_SPECIFIED)
        dataset_dict1 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values1, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict1, [axis_dict1])
        axis_values2 = index_start_extension_from + np.arange(length)
        dataset_values2 = 2 * np.ones(length)
        axis_dict2 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values2, AxisUnits.NOT_SPECIFIED)
        dataset_dict2 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values2, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict2, [axis_dict2])
        stored_axis = self.h5dmanager.get_axis("axis_group", "axis_name")
        stored_data = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(stored_axis, np.unique(np.concatenate([axis_values1, axis_values2])))
        np.testing.assert_array_equal(stored_data, np.concatenate([dataset_values1[:index_start_extension_from], dataset_values2]))

    def test_extend_shared_axis(self):
        length = 5
        axis_values1 = np.arange(length)
        dataset_values1 = np.ones(length)
        axis_dict1 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values1, AxisUnits.NOT_SPECIFIED)
        dataset_dict1 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name1", dataset_values1, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict1, [axis_dict1])
        stored_axis  = self.h5dmanager.get_axis("axis_group", "axis_name")
        stored_data1 = self.h5dmanager.get_dataset("dataset_group", "dataset_name1")
        axis_values2 = length + np.arange(length)
        dataset_values2 = 2 * np.ones(length)
        axis_dict2 = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values2, AxisUnits.NOT_SPECIFIED)
        dataset_dict2 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name2", dataset_values2, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict2, [axis_dict2])
        stored_axis  = self.h5dmanager.get_axis("axis_group", "axis_name")
        stored_data1 = self.h5dmanager.get_dataset("dataset_group", "dataset_name1")
        stored_data2 = self.h5dmanager.get_dataset("dataset_group", "dataset_name2")
        padded_dataset_values1 = np.concatenate([dataset_values1, np.full_like(dataset_values2, np.nan)])
        padded_dataset_values2 = np.concatenate([np.full_like(dataset_values1, np.nan), dataset_values2])
        np.testing.assert_array_equal(stored_axis, np.unique(np.concatenate([axis_values1, axis_values2])))
        np.testing.assert_array_equal(stored_data1, padded_dataset_values1)
        np.testing.assert_array_equal(stored_data2, padded_dataset_values2)

    def test_add_2d_dataset(self):
        length_rows = 50
        length_cols = 100
        dict_axis_rows = HDF5DataManager.create_dict_axis("axis_group", "axis_rows", np.arange(length_rows), AxisUnits.NOT_SPECIFIED)
        dict_axis_cols = HDF5DataManager.create_dict_axis("axis_group", "axis_cols", np.arange(length_cols), AxisUnits.NOT_SPECIFIED)
        dataset_values_in = np.random.rand(length_rows, length_cols)
        dict_dataset = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values_in, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dict_dataset, [dict_axis_rows, dict_axis_cols])
        dataset_values_read = self.h5dmanager.get_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(dataset_values_read, dataset_values_in)

if __name__ == '__main__':
    unittest.main()
