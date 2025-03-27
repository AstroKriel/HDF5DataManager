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
    def __init__(
            self,
            group, name, values,
            units = AxisUnits.NOT_SPECIFIED,
            notes = "",
        ):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        self.units  = units
        self.notes  = notes

    def add(self, values):
        ## the following is equivelant to np.sort(np.array(list(set(self.values) | set(values))))
        self.values = np.unique(np.concatenate((self.values, np.array(values))))

    def get_dict(self):
        return {
            "group"  : self.group,
            "name"   : self.name,
            "values" : self.values,
            "units"  : self.units.value,
            "notes"  : self.notes,
        }

    def __eq__(self, obj_axis):
        if isinstance(obj_axis, AxisObject):
            bool_properties_are_the_same = all([
                self.group == obj_axis.group,
                self.name  == obj_axis.name,
                np.array_equal(self.values, obj_axis.values)
            ])
            # print(f"axis `{self.group}/{self.name}` and `{obj_axis.group}/{obj_axis.name}` are {'the same' if bool_identical else 'different'}.") # for debugging
            return bool_properties_are_the_same
        return False

class DatasetObject:
    def __init__(
            self,
            group, name, values, list_axis_objs,
            units = DatasetUnits.NOT_SPECIFIED,
            notes = "",
        ):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        self.units  = units
        self.notes  = notes
        self.list_axis_objs = list_axis_objs

    def add(self, dataset_values_in, list_axis_values_in):
        self.reindex(
            dataset_values_in   = dataset_values_in,
            list_axis_values_in = list_axis_values_in,
        )

    def reindex(self, list_axis_values_in, dataset_values_in=None):
        num_axes_in = len(list_axis_values_in)
        num_dataset_dim = len(self.values.shape)
        if num_dataset_dim != num_axes_in:
            raise ValueError(f"Error: the number of provided axis do not match the dimensions of the existing dataset (`{num_axes_in}` != `{num_dataset_dim}`).")
        ## collect the stored axis values before updating
        list_axis_values_old = [
            np.array(obj_axis_old.values, copy=True)
            for obj_axis_old in self.list_axis_objs
        ]
        ## merge the input axis values into the stored axis
        [
            obj_axis_old.add(axis_values_in)
            for obj_axis_old, axis_values_in in zip(
                self.list_axis_objs,
                list_axis_values_in
            )
        ]
        ## resize the dataset dimensions to hold the `updated` = `old` + `in` data
        ## if no `in` data was provided, then the new space will be filled with `nan`s
        updated_dataset_shape = tuple(
            len(obj_axis_updated.values)
            for obj_axis_updated in self.list_axis_objs # this has been updated
        )
        dataset_values_updated = np.full(updated_dataset_shape, np.nan)
        dataset_indices_old = tuple(
            np.searchsorted(obj_axis_updated.values, old_axis_values)
            for obj_axis_updated, old_axis_values in zip(
                self.list_axis_objs,
                list_axis_values_old
            )
        )
        dataset_values_updated[np.ix_(*dataset_indices_old)] = self.values
        if dataset_values_in is not None:
            ## note: `in` may not necessarily be `new`
            ## where `old` == `in` -> overwrite
            ## where `old` != `in` -> merge
            dataset_indices_in = tuple(
                np.searchsorted(obj_axis_updated.values, new_axis_values)
                for obj_axis_updated, new_axis_values in zip(
                    self.list_axis_objs,
                    list_axis_values_in
                )
            )
            dataset_values_updated[np.ix_(*dataset_indices_in)] = dataset_values_in
        self.values = dataset_values_updated

    def get_dict(self):
        return {
            "group"  : self.group,
            "name"   : self.name,
            "values" : self.values,
            "units"  : self.units.value,
            "notes"  : self.notes,
            "list_axis_objs": [
                axis.get_dict()
                for axis in self.list_axis_objs
            ],
        }


class HDF5DataManager:
    def __init__(self):
        self.dict_axes_global = {}  # {group: {name: AxisObject}}
        self.dict_datasets = {}  # {group: {name: DatasetObject}}
        self.dict_axis_dependencies = {}  # { (axis_group, axis_name): [(dataset_group, dataset_name), ...] }

    def add(self, dict_dataset, list_axis_dicts):
        dataset_group  = dict_dataset.get("group")
        dataset_name   = dict_dataset.get("name")
        dataset_id     = (dataset_group, dataset_name)
        dataset_values = dict_dataset.get("values")
        dataset_units  = dict_dataset.get("units")
        dataset_notes  = dict_dataset.get("notes")
        self._validate_dimensions(dataset_values, list_axis_dicts)
        ## create the dataset group if it does not already exist
        if dataset_group not in self.dict_datasets:
            self.dict_datasets[dataset_group] = {}
        ## check whether the dataset needs to be initialised
        bool_init_dataset = dataset_name not in self.dict_datasets[dataset_group]
        ## if so, then axis objects will also need to be stored for initialisation
        list_axis_objs = []
        ## otherwise, only the input axis values need to be stored: to inform how the dataset should be reindexed
        list_axis_values = []
        for dict_axis in list_axis_dicts:
            axis_group  = dict_axis.get("group")
            axis_name   = dict_axis.get("name")
            axis_id     = (axis_group, axis_name)
            axis_values = dict_axis.get("values")
            axis_units  = dict_axis.get("units")
            axis_notes  = dict_axis.get("notes")
            ## make sure that the manager remembers that the dataset has a dependency on the axis
            if axis_id not in self.dict_axis_dependencies:
                self.dict_axis_dependencies[axis_id] = []
            if dataset_id not in self.dict_axis_dependencies[axis_id]:
                self.dict_axis_dependencies[axis_id].append(dataset_id)
            ## store information relevant for initialising or updating (merging + reindexing) the dataset
            if bool_init_dataset:
                obj_axis = AxisObject(
                    group  = axis_group,
                    name   = axis_name,
                    values = axis_values,
                    units  = axis_units,
                    notes  = axis_notes,
                )
                list_axis_objs.append(obj_axis)
            else: list_axis_values.append(axis_values)
            ## create the global version of the dataset group if it does not already exist
            if axis_group not in self.dict_axes_global:
                self.dict_axes_global[axis_group] = {}
            ## initialise the global axis object if it does not already exist
            if axis_name not in self.dict_axes_global[axis_group]:
                ## use a copy of the local axis object
                ## note: a deep-copy is necessary, so that the global values can be extended without affecting the local dataset axis
                self.dict_axes_global[axis_group][axis_name] = copy.deepcopy(obj_axis)
            ## make sure that the global axis object conatains the superset of all the values in the various instances of the same `axis_group/axis_name`
            else: self.dict_axes_global[axis_group][axis_name].add(axis_values)
        if bool_init_dataset:
            ## initialise the dataset object
            self.dict_datasets[dataset_group][dataset_name] = DatasetObject(
                group  = dataset_group,
                name   = dataset_name,
                values = dataset_values,
                units  = dataset_units,
                notes  = dataset_notes,
                list_axis_objs = list_axis_objs,
            )
        else:
            ## update the dataset shape and values; reindex using the input axis values to guide where `new` values should be inserted, or existing data should be overwritten
            self.dict_datasets[dataset_group][dataset_name].add(
                dataset_values_in   = dataset_values,
                list_axis_values_in = list_axis_values,
            )

    @staticmethod
    def _validate_dimensions(dataset_values, list_axis_dicts):
        list_errors = []
        ## ensure alignment between dataset and axes
        ## 1. make sure that the right number of axes have been provided
        if dataset_values.ndim != len(list_axis_dicts):
            list_errors.append(f"Dataset has {dataset_values.ndim} dimensions, but {len(list_axis_dicts)} axis objects are provided.")
        ## 2. make sure that each of the input dataset values has a corresponding axis value
        for axis_index, dict_axis in enumerate(list_axis_dicts):
            axis_values = dict_axis.get("values")
            if len(axis_values) != dataset_values.shape[axis_index]:
                list_errors.append(
                    f"Dimension {axis_index} of dataset values has shape {dataset_values.shape[axis_index]}, but axis `{dict_axis['name']}` has {len(axis_values)} values.")
        if len(list_errors) > 0: raise ValueError("\n".join(list_errors))

    def get_data_local(self, dataset_group, dataset_name):
        """get the dataset only where it has been defined."""
        if (dataset_group not in self.dict_datasets): return None
        if (dataset_name not in self.dict_datasets[dataset_group]): return None
        return self.dict_datasets[dataset_group][dataset_name].get_dict()

    def get_data_global(self, dataset_group, dataset_name):
        """get the dataset and indicate where data is not defined (compared to the global axis) with NaNs."""
        if dataset_group not in self.dict_datasets: return None
        if dataset_name not in self.dict_datasets[dataset_group]: return None
        ## make a deep copy so we do not modify the stored dataset
        obj_dataset_copy = copy.deepcopy(self.dict_datasets[dataset_group][dataset_name])
        ## build a list of global axis values not in the local axis. use this to reindex/reshape the dataset
        list_new_axis_values_from_global = []
        for obj_axis_local in obj_dataset_copy.list_axis_objs:
            obj_axis_global = self.dict_axes_global.get(obj_axis_local.group, {}).get(obj_axis_local.name)
            new_axis_values = np.array([])
            if obj_axis_global is not None: new_axis_values = np.setdiff1d(obj_axis_global.values, obj_axis_local.values)
            list_new_axis_values_from_global.append(new_axis_values)
        ## only reindex if at least one local axis is missing values in the global axis.
        if any(
                axis_values.size > 0
                for axis_values in list_new_axis_values_from_global
            ): obj_dataset_copy.reindex(list_axis_values_in=list_new_axis_values_from_global)
        return obj_dataset_copy.get_dict()

    @staticmethod
    def create_dict_axis(
            group, name, values,
            units = AxisUnits.NOT_SPECIFIED,
            notes = "",
        ):
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
        if np.any(np.isnan(values)):
            raise ValueError(f"Axis `{name}` contains NaN values.")
        if not np.all(np.diff(values) >= 0):
            raise ValueError("Axis values must be monotonically increasing.")
        if len(values) != len(np.unique(values)):
            raise ValueError("Axis values must be unique.")
        if not isinstance(units, AxisUnits):
            raise TypeError(f"Invalid axis unit: {units}. Must be an instance of AxisUnits.")
        if not isinstance(notes, str):
            raise TypeError("Notes must be a string.")
        return {
            "group"  : group,
            "name"   : name,
            "values" : values,
            "units"  : units,
            "notes"  : notes,
        }

    @staticmethod
    def create_dict_dataset(
            group, name, values,
            units = DatasetUnits.NOT_SPECIFIED,
            notes = "",
        ):
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
            "notes"  : notes,
        }

class TestHDF5DataManager(unittest.TestCase):

    def setUp(self):
        self.h5dmanager = HDF5DataManager()

    def test_invalid_axis_creation(self):
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_axis("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_axis("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            HDF5DataManager.create_dict_axis("valid_group", "valid_name", "invalid_values")  # no axis values
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_axis("valid_group", "valid_name", [[1, 2], [3, 4]])  # axis values are not a flat set of values
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_axis("valid_group", "valid_name", [2, 1, 4, 3])  # axis values are not monotonically increasing
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_axis("valid_group", "valid_name", [1, 1, 2, 2])  # axis values are not unique
        with self.assertRaises(TypeError):
            HDF5DataManager.create_dict_axis("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_invalid_dataset_creation(self):
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_dataset("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            HDF5DataManager.create_dict_dataset("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            HDF5DataManager.create_dict_dataset("valid_group", "valid_name", "invalid_values")  # no dataset values
        with self.assertRaises(TypeError):
            HDF5DataManager.create_dict_dataset("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_axis_creation(self):
        axis_values = np.arange(1000)
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED, notes="Test axis")
        self.assertEqual(axis_dict["group"], "axis_group")
        self.assertEqual(axis_dict["name"], "axis_name")
        np.testing.assert_array_equal(axis_dict["values"], axis_values)
        self.assertEqual(axis_dict["units"], AxisUnits.NOT_SPECIFIED)
        self.assertEqual(axis_dict["notes"], "Test axis")

    def test_dataset_creation(self):
        dataset_values = np.random.rand(1000) * 100
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.DIMENSIONLESS, notes="Test dataset")
        self.assertEqual(dataset_dict["group"], "dataset_group")
        self.assertEqual(dataset_dict["name"], "dataset_name")
        np.testing.assert_array_equal(dataset_dict["values"], dataset_values)
        self.assertEqual(dataset_dict["units"], DatasetUnits.DIMENSIONLESS)
        self.assertEqual(dataset_dict["notes"], "Test dataset")

    def test_add_data(self):
        length = 100
        axis_values = np.arange(length)
        dataset_values = np.random.rand(length)
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict, [axis_dict])
        dict_dataset = self.h5dmanager.get_data_local("dataset_group", "dataset_name")
        self.assertIsNotNone(dict_dataset)
        stored_data = dict_dataset["values"]
        np.testing.assert_array_equal(stored_data, dataset_values)

    def test_get_nonexistent_datasets(self):
        self.assertIsNone(self.h5dmanager.get_data_local("nonexistent_group", "nonexistent_name"))
        self.assertIsNone(self.h5dmanager.get_data_global("nonexistent_group", "nonexistent_name"))

    def test_adding_multiple_datasets_with_a_shared_axis(self):
        length_1 = 50
        length_2 = 20
        axis_values_1 = np.arange(length_1)
        axis_values_2 = np.arange(length_2)
        dataset_values_1 = np.random.rand(length_1)
        dataset_values_2 = np.random.rand(length_1, length_2)
        axis_dict_1 = HDF5DataManager.create_dict_axis("axis_group_1", "axis_name_1", axis_values_1, AxisUnits.NOT_SPECIFIED)
        axis_dict_2 = HDF5DataManager.create_dict_axis("axis_group_2", "axis_name_2", axis_values_2, AxisUnits.NOT_SPECIFIED)
        dataset_dict_1 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_1", dataset_values_1, DatasetUnits.NOT_SPECIFIED)
        dataset_dict_2 = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_2", dataset_values_2, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict_1, [axis_dict_1])
        self.h5dmanager.add(dataset_dict_2, [axis_dict_1, axis_dict_2])
        dict_dataset_1 = self.h5dmanager.get_data_local("dataset_group", "dataset_1")
        dict_dataset_2 = self.h5dmanager.get_data_local("dataset_group", "dataset_2")
        self.assertIsNotNone(dict_dataset_1)
        self.assertIsNotNone(dict_dataset_2)
        np.testing.assert_array_equal(dict_dataset_1["values"], dataset_values_1)
        np.testing.assert_array_equal(dict_dataset_2["values"], dataset_values_2)

    def test_extending_dataset(self):
        axis_values = np.array([0, 1, 2])
        dataset_values = np.array([10, 20, 30])
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict, [axis_dict])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", new_axis_values, AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group", "dataset_name", new_dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(new_dataset_dict, [new_axis_dict])
        global_data = self.h5dmanager.get_data_global("dataset_group", "dataset_name")
        self.assertIsNotNone(global_data)
        expected_values = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(global_data["values"], expected_values)

    def test_reindexing_dataset_to_get_values_with_global_axis_values(self):
        axis_values = np.array([0, 1, 2])
        dataset_values = np.array([10, 20, 30])
        axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group_1", "dataset_name_1", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(dataset_dict, [axis_dict])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = HDF5DataManager.create_dict_axis("axis_group", "axis_name", new_axis_values, AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = HDF5DataManager.create_dict_dataset("dataset_group_2", "dataset_name_2", new_dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.h5dmanager.add(new_dataset_dict, [new_axis_dict])
        global_data = self.h5dmanager.get_data_global("dataset_group_1", "dataset_name_1")
        self.assertIsNotNone(global_data)
        expected_values = np.array([10, 20, 30, np.nan, np.nan])
        np.testing.assert_array_equal(global_data["values"], expected_values)

    def test_dataset_reindexing_expands_correctly(self):
        axis_values = np.array([3, 4])
        dataset_values = np.array([30, 40])
        axis = AxisObject("group1", "axis1", axis_values)
        dataset = DatasetObject("group1", "dataset1", dataset_values, [axis])
        axis_values_global = np.array([0, 1, 2, 3, 4, 5, 6])
        dataset.reindex([axis_values_global])
        self.assertEqual(dataset.values.shape[0], len(axis_values_global))
        np.testing.assert_array_equal(dataset.values, [np.nan, np.nan, np.nan, 30, 40, np.nan, np.nan])

    def test_dataset_overwrite_with_new_data(self):
        axis_values = np.array([1, 2])
        dataset_values = np.array([10, 20])
        axis = AxisObject("group1", "axis1", axis_values)
        dataset = DatasetObject("group1", "dataset1", dataset_values, [axis])
        new_axis_values = np.array([0, 1, 2, 3])
        new_dataset_values = np.array([0, 11, 22, 30])
        dataset.reindex([new_axis_values], new_dataset_values)
        expected_values = np.array([0, 11, 22, 30])
        np.testing.assert_array_equal(dataset.values, expected_values)

if __name__ == "__main__":
    unittest.main()
