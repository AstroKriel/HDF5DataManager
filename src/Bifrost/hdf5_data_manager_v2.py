import os
import copy
import h5py
import json
import tempfile
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
            units      = AxisUnits.NOT_SPECIFIED.value,
            notes      = "",
            global_ref = None,
        ):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        ## local properties should be private
        units = self._cast_unit(units)
        if not isinstance(notes, str):
            raise TypeError("notes need to be a string")
        self._units = units
        self._notes = notes
        self._global_ref = global_ref

    @property
    def units(self):
        if self._global_ref is not None:
            return self._global_ref.units
        return self._units

    @units.setter
    def units(self, units):
        units = self._cast_unit(units)
        if self._global_ref is not None:
            self._global_ref.units = units
        else: self._units = units

    @property
    def notes(self):
        if self._global_ref is not None:
            return self._global_ref.notes
        return self._notes

    @notes.setter
    def notes(self, notes):
        if not isinstance(notes, str):
            raise TypeError("notes need to be a string.")
        if self._global_ref is not None:
            self._global_ref.notes = notes
        else: self._notes = notes

    @staticmethod
    def _validate_inputs(group, name, values, units, notes):
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Axis group must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Axis name must be a non-empty string.")
        if not isinstance(values, (list, np.ndarray)):
            raise TypeError("Axis values must be a list or numpy array.")
        values = np.array(values)
        if values.size == 0:
            raise ValueError("Axis values cannot be empty.")
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
            raise TypeError(f"Invalid unit: {units}. Must be a string or an element in AxisUnits.")
        if not isinstance(notes, str):
            raise TypeError("Notes must be a string.")

    @staticmethod
    def _cast_unit(units):
        if not isinstance(units, (str, AxisUnits)):
            raise TypeError("units need to either be a string or an element of AxisUnits.")
        if isinstance(units, AxisUnits): units = units.value
        return units

    @staticmethod
    def create_dict_inputs(
            group, name, values,
            units = AxisUnits.NOT_SPECIFIED,
            notes = "",
        ):
        AxisObject._validate_inputs(group, name, values, units, notes)
        units = AxisObject._cast_unit(units)
        return {
            "group"  : group,
            "name"   : name,
            "values" : values,
            "units"  : units,
            "notes"  : notes,
        }

    def add(self, values):
        ## merge the new + unique values into the existing set
        ## note: the following is equivelant to `np.sort(np.array(list(set(self.values) | set(values))))`
        self.values = np.unique(np.concatenate((self.values, np.array(values))))

    def get_dict(self):
        return {
            "group"  : self.group,
            "name"   : self.name,
            "values" : self.values,
            "units"  : self.units,
            "notes"  : self.notes,
        }

    def __eq__(self, obj_axis):
        if isinstance(obj_axis, AxisObject):
            bool_properties_are_the_same = all([
                self.group == obj_axis.group,
                self.name  == obj_axis.name,
                np.array_equal(self.values, obj_axis.values)
            ])
            return bool_properties_are_the_same
        return False

class DatasetObject:
    def __init__(
            self,
            group, name, values, list_axis_objs,
            units = DatasetUnits.NOT_SPECIFIED.value,
            notes = "",
        ):
        self.group  = group
        self.name   = name
        self.values = np.array(values)
        ## store units as a string
        units = self._cast_unit(units)
        self.units = units
        self.notes = notes
        self.list_axis_objs = list_axis_objs

    @staticmethod
    def _validate_inputs(group, name, values, units, notes):
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Dataset group must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")
        if not isinstance(values, (list, np.ndarray)):
            raise TypeError("Dataset values must be a list or numpy array.")
        values = np.array(values)
        if values.size == 0:
            raise ValueError("Dataset values cannot be empty.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Dataset values must be numeric (either integers or floats).")
        if not isinstance(units, DatasetUnits):
            raise TypeError(f"Invalid dataset unit: {units}. Must be an instance of DatasetUnits.")
        if not isinstance(notes, str):
            raise TypeError("Notes must be a string.")

    @staticmethod
    def _cast_unit(units):
        if not isinstance(units, (str, DatasetUnits)):
            raise TypeError("units need to either be a string or an element of DatasetUnits.")
        if isinstance(units, DatasetUnits): units = units.value
        return units

    @staticmethod
    def create_dict_inputs(
            group, name, values,
            units = DatasetUnits.NOT_SPECIFIED,
            notes = "",
        ):
        DatasetObject._validate_inputs(group, name, values, units, notes)
        units = DatasetObject._cast_unit(units)
        return {
            "group"  : group,
            "name"   : name,
            "values" : np.array(values),
            "units"  : units,
            "notes"  : notes,
        }

    def get_dict(self):
        return {
            "group"  : self.group,
            "name"   : self.name,
            "values" : self.values,
            "units"  : self.units,
            "notes"  : self.notes,
            "list_axis_dicts": [
                obj_axis.get_dict()
                for obj_axis in self.list_axis_objs
            ],
        }

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

class HDF5DataManager:
    def __init__(self):
        self.dict_global_axes = {}  # { axes_group: { axis_name: AxisObject, ... }, ... }
        self.dict_datasets = {}  # { dataset_group: { dataset_name: DatasetObject, ... }, ... }
        self.dict_axis_dependencies = {}  # { (axis_group, axis_name): [ (dataset_group, dataset_name), ... ], ... }

    @staticmethod
    def validate_h5file_structure(file_path: str) -> bool:
        """
        Validates the structure of an HDF5 file to ensure it meets the expected format.

        Args:
            file_path (str): Path to the HDF5 file to validate.

        Returns:
            bool: True if the file structure is valid, False otherwise.
        """
        try:
            with h5py.File(file_path, "r") as f:
                # Check for required top-level groups
                if "global_axes" not in f or "datasets" not in f:
                    return False
                # Validate global_axes structure
                for axis_group in f["global_axes"].values():
                    if not isinstance(axis_group, h5py.Group):
                        return False
                    for axis in axis_group.values():
                        if "units" not in axis.attrs or "notes" not in axis.attrs:
                            return False
                # Validate datasets structure
                for dataset_group in f["datasets"].values():
                    if not isinstance(dataset_group, h5py.Group):
                        return False
                    for dataset in dataset_group.values():
                        if "values" not in dataset or "local_axes" not in dataset:
                            return False
                        if "units" not in dataset.attrs or "notes" not in dataset.attrs:
                            return False
                return True
        except Exception as e:
            return False

    @classmethod
    def load_hdf5_file(cls, file_path):
        if not HDF5DataManager.validate_h5file_structure(file_path):
            raise ValueError(f"Invalid HDF5 file structure: {file_path}")
        obj_h5dm = cls() # initialise an instance of HDF5DataManager
        with h5py.File(file_path, "r") as h5_file:
            ## Load global axes
            if "global_axes" in h5_file:
                for axis_group, h5_global_axes_group in h5_file["global_axes"].items():
                    obj_h5dm.dict_global_axes[axis_group] = {}
                    for axis_name, h5_global_axis in h5_global_axes_group.items():
                        obj_h5dm.dict_global_axes[axis_group][axis_name] = AxisObject(
                            group  = axis_group,
                            name   = axis_name,
                            values = np.array(h5_global_axis[:]),
                            units  = h5_global_axis.attrs.get("units", ""),
                            notes  = h5_global_axis.attrs.get("notes", ""),
                        )
                        axis_id = (axis_group, axis_name)
                        str_axis_dependencies = h5_global_axis.attrs.get("dependencies", "[]")
                        obj_h5dm.dict_axis_dependencies[axis_id] = [
                            tuple(dependency.split("/"))
                            for dependency in json.loads(str_axis_dependencies)
                        ]
            ## Load datasets
            if "datasets" in h5_file:
                for dataset_group, h5_datasets in h5_file["datasets"].items():
                    obj_h5dm.dict_datasets[dataset_group] = {}
                    for dataset_name, h5_dataset in h5_datasets.items():
                        if not isinstance(h5_dataset, h5py.Group): continue
                        ## Load dataset values and metadata
                        dataset_values = np.array(h5_dataset["values"])
                        dataset_units  = h5_dataset.attrs.get("units", "")
                        dataset_notes  = h5_dataset.attrs.get("notes", "")
                        ## Load local axes
                        list_axis_objs = []
                        if "local_axes" in h5_dataset:
                            h5_local_axes_group = h5_dataset["local_axes"]
                            for axis_name in h5_local_axes_group:
                                h5_local_axis = h5_local_axes_group[axis_name]
                                axis_group    = h5_local_axis.attrs["group"]
                                axis_name     = h5_local_axis.attrs["name"]
                                axis_values   = np.array(h5_local_axis["values"])
                                ## Retrieve global reference, if available
                                global_ref = obj_h5dm.dict_global_axes.get(axis_group, {}).get(axis_name, None)
                                ## Create local axis with global_ref
                                axis_obj = AxisObject(
                                    group  = axis_group,
                                    name   = axis_name,
                                    values = axis_values,
                                    units  = "",  # no units stored for local axes
                                    notes  = "",  # no notes stored for local axes
                                    global_ref = global_ref,  # set global reference
                                )
                                list_axis_objs.append(axis_obj)
                        ## Store dataset
                        obj_h5dm.dict_datasets[dataset_group][dataset_name] = DatasetObject(
                            group  = dataset_group,
                            name   = dataset_name,
                            values = dataset_values,
                            units  = dataset_units,
                            notes  = dataset_notes,
                            list_axis_objs = list_axis_objs,
                        )
        return obj_h5dm

    def save_hdf5_file(self, file_path):
        with h5py.File(file_path, "w") as h5_file:
            ## save global axes
            h5_global_axes = h5_file.create_group("global_axes")
            for axis_group, dict_axes_group in self.dict_global_axes.items():
                h5_global_axes_group = h5_global_axes.create_group(axis_group)
                for axis_name, obj_axis_global in dict_axes_group.items():
                    ## store global axis values: super-set of all the values in the various instances of this axis
                    h5_global_axis = h5_global_axes_group.create_dataset(axis_name, data=np.array(obj_axis_global.values))
                    units = AxisObject._cast_unit(obj_axis_global.units)
                    h5_global_axis.attrs["units"] = units
                    h5_global_axis.attrs["notes"] = obj_axis_global.notes
                    ## store dataset dependencies: list of dataset paths that use this axis
                    axis_id = (axis_group, axis_name)
                    if axis_id in self.dict_axis_dependencies:
                        list_axis_dependencies = [
                            f"{dataset_group}/{dataset_name}"
                            for dataset_group, dataset_name in self.dict_axis_dependencies[axis_id]
                        ]
                        h5_global_axis.attrs["dependencies"] = json.dumps(list_axis_dependencies) # store as JSON string
            ## save datasets
            h5_datasets = h5_file.create_group("datasets")
            for dataset_group, datasets in self.dict_datasets.items():
                h5_datasets_group = h5_datasets.create_group(dataset_group)
                for dataset_name, obj_dataset in datasets.items():
                    h5_dataset = h5_datasets_group.create_group(dataset_name)
                    h5_dataset.create_dataset("values", data=np.array(obj_dataset.values))
                    units = DatasetObject._cast_unit(obj_dataset.units)
                    h5_dataset.attrs["units"] = units
                    h5_dataset.attrs["notes"] = obj_dataset.notes
                    h5_local_axes = h5_dataset.create_group("local_axes")
                    for axis_index, obj_axis_local in enumerate(obj_dataset.list_axis_objs):
                        h5_local_axis = h5_local_axes.create_group(f"axis_{axis_index}")
                        h5_local_axis.attrs["group"] = obj_axis_local.group
                        h5_local_axis.attrs["name"] = obj_axis_local.name
                        h5_local_axis.create_dataset("values", data=np.array(obj_axis_local.values))

    def add(self, dict_dataset, list_axis_dicts):
        dataset_group  = dict_dataset.get("group")
        dataset_name   = dict_dataset.get("name")
        dataset_id     = (dataset_group, dataset_name)
        dataset_values = dict_dataset.get("values")
        dataset_units  = dict_dataset.get("units")
        if isinstance(dataset_units, DatasetUnits): dataset_units = dataset_units.value
        dataset_notes  = dict_dataset.get("notes")
        HDF5DataManager._validate_dimensions(dataset_values, list_axis_dicts)
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
            if isinstance(axis_units, AxisUnits): axis_units = axis_units.value
            axis_notes  = dict_axis.get("notes")
            ## make sure that the manager remembers that the dataset has a dependency on the axis
            if axis_id not in self.dict_axis_dependencies:
                self.dict_axis_dependencies[axis_id] = []
            if dataset_id not in self.dict_axis_dependencies[axis_id]:
                self.dict_axis_dependencies[axis_id].append(dataset_id)
            ## store information relevant for initialising or updating (merging + reindexing) the dataset
            if bool_init_dataset:
                obj_axis_local = AxisObject(
                    group  = axis_group,
                    name   = axis_name,
                    values = axis_values,
                    units  = axis_units,
                    notes  = axis_notes,
                )
                list_axis_objs.append(obj_axis_local)
            else: list_axis_values.append(axis_values)
            ## create the global version of the dataset group if it does not already exist
            if axis_group not in self.dict_global_axes:
                self.dict_global_axes[axis_group] = {}
            ## initialise the global axis object if it does not already exist
            if axis_name not in self.dict_global_axes[axis_group]:
                ## use a copy of the local axis object
                ## note: a deep-copy is necessary, so that the global values can be extended without affecting the local dataset axis
                self.dict_global_axes[axis_group][axis_name] = copy.deepcopy(obj_axis_local)
                ## store a reference of the global axis
                obj_axis_local._global_ref = self.dict_global_axes[axis_group][axis_name]
            ## make sure that the global axis object conatains the superset of all the values in the various instances of the same `axis_group/axis_name`
            else:
                obj_axis_global = self.dict_global_axes[axis_group][axis_name]
                obj_axis_global.add(axis_values)
                if axis_units != AxisUnits.NOT_SPECIFIED.value: obj_axis_global.units = axis_units
                if len(axis_notes) > 0: obj_axis_global.notes = axis_notes
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
            list_errors.append(f"Dataset has {dataset_values.ndim} dimensions, but only {len(list_axis_dicts)} axis object(s) has been provided.")
        ## 2. make sure that each of the input dataset values has a corresponding axis value
        for axis_index, dict_axis in enumerate(list_axis_dicts):
            axis_values = dict_axis.get("values")
            if len(axis_values) != dataset_values.shape[axis_index]:
                list_errors.append(
                    f"Dimension {axis_index} of dataset values has shape {dataset_values.shape[axis_index]}, but axis `{dict_axis['name']}` has {len(axis_values)} values.")
        if len(list_errors) > 0: raise ValueError("\n".join(list_errors))

    def update_global_axis_metadata(self, axis_group, axis_name, units=None, notes=None):
        if axis_group not in self.dict_global_axes:
            raise ValueError(f"Global axis group '{axis_group}' not found.")
        if axis_name not in self.dict_global_axes[axis_group]:
            raise ValueError(f"Global axis '{axis_name}' not found in group '{axis_group}'.")
        obj_axis_global = self.dict_global_axes[axis_group][axis_name]
        if units is not None:
            if not isinstance(units, (str, AxisUnits)):
                raise TypeError("new_units must be either a string or an element of AxisUnits.")
            obj_axis_global.units = units
        if notes is not None:
            if not isinstance(notes, str):
                raise TypeError("new_notes must be a string.")
            obj_axis_global.notes = notes

    def get_local_dataset(self, dataset_group, dataset_name):
        """get the dataset only where it has been defined."""
        if (dataset_group not in self.dict_datasets): return None
        if (dataset_name not in self.dict_datasets[dataset_group]): return None
        return self.dict_datasets[dataset_group][dataset_name].get_dict()

    def get_global_dataset(self, dataset_group, dataset_name):
        """get the dataset and indicate where data is not defined (compared to the global axis) with NaNs."""
        if dataset_group not in self.dict_datasets: return None
        if dataset_name not in self.dict_datasets[dataset_group]: return None
        ## make a deep copy so we do not modify the stored dataset
        obj_dataset_copy = copy.deepcopy(self.dict_datasets[dataset_group][dataset_name])
        ## build a list of global axis values not in the local axis. use this to reindex/reshape the dataset
        list_new_axis_values_from_global = []
        for obj_axis_local in obj_dataset_copy.list_axis_objs:
            obj_axis_global = self.dict_global_axes.get(obj_axis_local.group, {}).get(obj_axis_local.name)
            new_axis_values = np.array([])
            if obj_axis_global is not None: new_axis_values = np.setdiff1d(obj_axis_global.values, obj_axis_local.values)
            list_new_axis_values_from_global.append(new_axis_values)
        ## only reindex if at least one local axis is missing values in the global axis.
        if any(
                axis_values.size > 0
                for axis_values in list_new_axis_values_from_global
            ): obj_dataset_copy.reindex(list_axis_values_in=list_new_axis_values_from_global)
        return obj_dataset_copy.get_dict()

class TestHDF5DataManager(unittest.TestCase):

    def setUp(self):
        self.obj_h5dm = HDF5DataManager()

    def test_invalid_axis_creation(self):
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            AxisObject.create_dict_inputs("valid_group", "valid_name", "invalid_values")  # no axis values
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("valid_group", "valid_name", [[1, 2], [3, 4]])  # axis values are not a flat set of values
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("valid_group", "valid_name", [2, 1, 4, 3])  # axis values are not monotonically increasing
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("valid_group", "valid_name", [1, 1, 2, 2])  # axis values are not unique
        with self.assertRaises(TypeError):
            AxisObject.create_dict_inputs("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_invalid_dataset_creation(self):
        with self.assertRaises(ValueError):
            DatasetObject.create_dict_inputs("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            DatasetObject.create_dict_inputs("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("valid_group", "valid_name", "invalid_values")  # no dataset values
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_axis_creation(self):
        axis_values = np.arange(1000)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED, notes="Test axis")
        self.assertEqual(dict_axis["group"], "axis_group")
        self.assertEqual(dict_axis["name"], "axis_name")
        np.testing.assert_array_equal(dict_axis["values"], axis_values)
        self.assertEqual(dict_axis["units"], AxisUnits.NOT_SPECIFIED.value)
        self.assertEqual(dict_axis["notes"], "Test axis")

    def test_dataset_creation(self):
        dataset_values = np.random.rand(1000) * 100
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.DIMENSIONLESS, notes="Test dataset")
        self.assertEqual(dict_dataset["group"], "dataset_group")
        self.assertEqual(dict_dataset["name"], "dataset_name")
        np.testing.assert_array_equal(dict_dataset["values"], dataset_values)
        self.assertEqual(dict_dataset["units"], DatasetUnits.DIMENSIONLESS.value)
        self.assertEqual(dict_dataset["notes"], "Test dataset")

    def test_add_data(self):
        length = 100
        axis_values = np.arange(length)
        dataset_values = np.random.rand(length)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        dict_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        self.assertIsNotNone(dict_dataset)
        stored_data = dict_dataset["values"]
        np.testing.assert_array_equal(stored_data, dataset_values)

    def test_get_nonexistent_datasets(self):
        self.assertIsNone(self.obj_h5dm.get_local_dataset("nonexistent_group", "nonexistent_name"))
        self.assertIsNone(self.obj_h5dm.get_global_dataset("nonexistent_group", "nonexistent_name"))

    def test_adding_multiple_datasets_with_a_shared_axis(self):
        length_1 = 50
        length_2 = 20
        axis_values_1 = np.arange(length_1)
        axis_values_2 = np.arange(length_2)
        dataset_values_1 = np.random.rand(length_1)
        dataset_values_2 = np.random.rand(length_1, length_2)
        axis_dict_1 = AxisObject.create_dict_inputs("axis_group_1", "axis_name_1", axis_values_1, AxisUnits.NOT_SPECIFIED)
        axis_dict_2 = AxisObject.create_dict_inputs("axis_group_2", "axis_name_2", axis_values_2, AxisUnits.NOT_SPECIFIED)
        dataset_dict_1 = DatasetObject.create_dict_inputs("dataset_group", "dataset_1", dataset_values_1, DatasetUnits.NOT_SPECIFIED)
        dataset_dict_2 = DatasetObject.create_dict_inputs("dataset_group", "dataset_2", dataset_values_2, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dataset_dict_1, [axis_dict_1])
        self.obj_h5dm.add(dataset_dict_2, [axis_dict_1, axis_dict_2])
        dict_dataset_1 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_1")
        dict_dataset_2 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_2")
        self.assertIsNotNone(dict_dataset_1)
        self.assertIsNotNone(dict_dataset_2)
        np.testing.assert_array_equal(dict_dataset_1["values"], dataset_values_1)
        np.testing.assert_array_equal(dict_dataset_2["values"], dataset_values_2)

    def test_extending_dataset(self):
        axis_values = np.array([0, 1, 2])
        dataset_values = np.array([10, 20, 30])
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = AxisObject.create_dict_inputs("axis_group", "axis_name", new_axis_values, AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", new_dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(new_dataset_dict, [new_axis_dict])
        global_data = self.obj_h5dm.get_global_dataset("dataset_group", "dataset_name")
        self.assertIsNotNone(global_data)
        expected_values = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(global_data["values"], expected_values)

    def test_reindexing_dataset_to_get_values_with_global_axis_values(self):
        axis_values = np.array([0, 1, 2])
        dataset_values = np.array([10, 20, 30])
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group_1", "dataset_name_1", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = AxisObject.create_dict_inputs("axis_group", "axis_name", new_axis_values, AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = DatasetObject.create_dict_inputs("dataset_group_2", "dataset_name_2", new_dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(new_dataset_dict, [new_axis_dict])
        global_data = self.obj_h5dm.get_global_dataset("dataset_group_1", "dataset_name_1")
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

    def test_add_and_update_metadata(self):
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2])
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30])
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        # Update metadata
        self.obj_h5dm.update_global_axis_metadata("axis_group", "axis_name", units=AxisUnits.T_TURB, notes="Updated notes")
        updated_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["units"], AxisUnits.T_TURB.value)
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["notes"], "Updated notes")

    def test_extend_existing_dataset(self):
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2])
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30])
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        # Extend dataset
        new_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [3, 4])
        new_data = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [40, 50])
        self.obj_h5dm.add(new_data, [new_axis])
        extended_dataset = self.obj_h5dm.get_global_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(extended_dataset["values"], [10, 20, 30, 40, 50])

    # Edge case 1: Empty Axis Values
    def test_empty_axis_values(self):
        empty_axis_values = []
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("axis_group", "axis_name", empty_axis_values, AxisUnits.NOT_SPECIFIED)

    # Edge case 3: Adding dataset with mismatched axis shapes
    def test_mismatched_axis_and_dataset_shapes(self):
        axis_values = np.arange(3)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_values = np.random.rand(4)  # Dataset shape does not match axis shape
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [dict_axis])

    # Edge case 4: Axis values with NaN values
    def test_axis_values_with_nan(self):
        axis_values = np.array([1, 2, np.nan, 4])
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)

    # Edge case 5: Invalid axis unit type
    def test_invalid_axis_unit_type(self):
        axis_values = np.arange(3)
        with self.assertRaises(TypeError):
            AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, "invalid_unit")

    # Edge case 6: Invalid dataset unit type
    def test_invalid_dataset_unit_type(self):
        dataset_values = np.random.rand(3)
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, "invalid_unit")

    # Edge case 7: Adding new axis values that overlap
    def test_adding_overlap_axis_values(self):
        axis_values_1 = np.array([1, 2, 3])
        axis_values_2 = np.array([2, 3, 4])
        dataset_values = np.random.rand(3)
        axis_dict_1 = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values_1, AxisUnits.NOT_SPECIFIED)
        axis_dict_2 = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values_2, AxisUnits.NOT_SPECIFIED)
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [axis_dict_1])
        self.obj_h5dm.add(dict_dataset, [axis_dict_2])
        # Check that both axis values are merged correctly without duplication
        dict_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        axis_values_expected = np.unique(np.concatenate((axis_values_1, axis_values_2)))
        np.testing.assert_array_equal(dict_dataset["list_axis_dicts"][0]["values"], axis_values_expected)

    # Edge case 9: Adding duplicate dataset with identical axis
    def test_duplicate_dataset_with_identical_axis(self):
        axis_values = np.arange(3)
        dataset_values = np.random.rand(3)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        # Add dataset once
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        dict_dataset_1 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        self.assertIsNotNone(dict_dataset_1)
        # Add identical dataset again (should not create a new dataset)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        dict_dataset_2 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(dict_dataset_1["values"], dict_dataset_2["values"])

    # Edge case 10: Adding dataset with missing axis values
    def test_missing_axis_values(self):
        dataset_values = np.random.rand(3)  # Dataset does not have missing values in this case
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        # Add the dataset with a missing axis
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [])

    # Edge case 11: Adding datasets with different units
    def test_adding_datasets_with_different_units(self):
        axis_values = np.arange(3)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        # Dataset with dimensionless units
        dataset_values_1 = np.random.rand(3)
        dataset_dict_1 = DatasetObject.create_dict_inputs("dataset_group", "dataset_1", dataset_values_1, DatasetUnits.DIMENSIONLESS)
        # Dataset with different units
        dataset_values_2 = np.random.rand(3)
        dataset_dict_2 = DatasetObject.create_dict_inputs("dataset_group", "dataset_2", dataset_values_2, DatasetUnits.NOT_SPECIFIED)
        # Adding datasets with different units
        self.obj_h5dm.add(dataset_dict_1, [dict_axis])
        self.obj_h5dm.add(dataset_dict_2, [dict_axis])
        dict_dataset_1 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_1")
        dict_dataset_2 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_2")
        self.assertIsNotNone(dict_dataset_1)
        self.assertIsNotNone(dict_dataset_2)
    
    def test_global_axis_inheritance(self):
        # Update the global axis metadata using the update function
        self.dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=AxisUnits.DIMENSIONLESS, notes="original axis")
        self.dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=DatasetUnits.DIMENSIONLESS, notes="test dataset")
        self.obj_h5dm.add(self.dict_dataset, [self.dict_axis])
        new_units = AxisUnits.T_TURB
        new_notes = "updated global axis notes"
        self.obj_h5dm.update_global_axis_metadata("axis_group", "axis_name", units=new_units, notes=new_notes)
        # Get the local dataset and check that its local axis reflects the change.
        dict_local_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        dict_local_axis = dict_local_dataset["list_axis_dicts"][0]
        # The local axis should now use the global axis metadata.
        self.assertEqual(dict_local_axis["units"], new_units.value)
        self.assertEqual(dict_local_axis["notes"], new_notes)

class TestHDF5DataManagerSaveLoad(unittest.TestCase):

    def test_valid_file_structure(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=AxisUnits.T_TURB, notes="original axis")
            dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=DatasetUnits.DIMENSIONLESS, notes="test dataset")
            obj_h5dm_save = HDF5DataManager()
            obj_h5dm_save.add(dict_dataset, [dict_axis])
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            self.assertTrue(HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_invalid_file_structure(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group("invalid_group")
            self.assertFalse(HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_nonexistent_file(self):
        non_existent_file = "non_existent_file.h5"
        self.assertFalse(HDF5DataManager.validate_h5file_structure(non_existent_file))

    def test_missing_required_groups(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group("global_axes")  # Missing "datasets" group
            self.assertFalse(HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_invalid_dataset_structure(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group("global_axes")
                datasets = f.create_group("datasets")
                dataset_group = datasets.create_group("dataset_group")
                dataset = dataset_group.create_group("dataset_name")
                dataset.create_dataset("values", data=[1, 2, 3])
                # Missing "local_axes" group
            self.assertFalse(HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_save_and_load_hdf5_file(self):
        # Save the current manager to a temporary HDF5 file
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=AxisUnits.T_TURB, notes="original axis")
            dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=DatasetUnits.DIMENSIONLESS, notes="test dataset")
            obj_h5dm_save = HDF5DataManager()
            obj_h5dm_save.add(dict_dataset, [dict_axis])
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            obj_h5dm_load = HDF5DataManager.load_hdf5_file(tmp_filename)
            dict_local_dataset_saved = obj_h5dm_save.get_local_dataset("dataset_group", "dataset_name")
            dict_local_dataset_loaded = obj_h5dm_load.get_local_dataset("dataset_group", "dataset_name")
            self.assertIsNotNone(dict_local_dataset_loaded)
            np.testing.assert_array_equal(dict_local_dataset_loaded["values"], dict_local_dataset_saved["values"])
            self.assertEqual(dict_local_dataset_loaded["units"], dict_local_dataset_saved["units"])
            self.assertEqual(dict_local_dataset_loaded["notes"], dict_local_dataset_saved["notes"])
            # Verify that global axis metadata is preserved.
            dict_global_axis_saved  = obj_h5dm_save.dict_global_axes["axis_group"]["axis_name"]
            dict_global_axis_loaded = obj_h5dm_load.dict_global_axes["axis_group"]["axis_name"]
            np.testing.assert_array_equal(dict_global_axis_loaded.values, dict_global_axis_saved.values)
            self.assertEqual(dict_global_axis_loaded.units, dict_global_axis_saved.units)
            self.assertEqual(dict_global_axis_loaded.notes, dict_global_axis_saved.notes)
            # Verify that the local axis dictionaries in the dataset match the global axis metadata.
            # Here we assume that when saving/loading, the local axis info is stored in "list_axis_dicts".
            dict_local_axis_saved  = dict_local_dataset_saved["list_axis_dicts"][0]
            dict_local_axis_loaded = dict_local_dataset_loaded["list_axis_dicts"][0]
            np.testing.assert_array_equal(dict_local_axis_saved["values"], dict_local_axis_loaded["values"])
            # Since local axes now derive from the global axis, their units/notes should match.
            self.assertEqual(dict_local_axis_loaded["units"], dict_global_axis_saved.units)
            self.assertEqual(dict_local_axis_loaded["notes"], dict_global_axis_saved.notes)
        finally:
            os.remove(tmp_filename)

    def test_save_load_multiple_datasets(self):
        # Add multiple datasets
        axis1 = AxisObject.create_dict_inputs("group1", "axis1", [0, 1, 2])
        axis2 = AxisObject.create_dict_inputs("group2", "axis2", [3, 4, 5])
        dataset1 = DatasetObject.create_dict_inputs("group1", "dataset1", [10, 20, 30])
        dataset2 = DatasetObject.create_dict_inputs("group2", "dataset2", [40, 50, 60])
        obj_h5dm_save = HDF5DataManager()
        obj_h5dm_save.add(dataset1, [axis1])
        obj_h5dm_save.add(dataset2, [axis2])
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            obj_h5dm_load = HDF5DataManager.load_hdf5_file(tmp_filename)
            loaded_dataset1 = obj_h5dm_load.get_local_dataset("group1", "dataset1")
            loaded_dataset2 = obj_h5dm_load.get_local_dataset("group2", "dataset2")
            np.testing.assert_array_equal(loaded_dataset1["values"], [10, 20, 30])
            np.testing.assert_array_equal(loaded_dataset2["values"], [40, 50, 60])
        finally:
            os.remove(tmp_filename)

    def test_file_structure_validation(self):
        # Create a valid file
        obj_h5dm = HDF5DataManager()
        axis = AxisObject.create_dict_inputs("group", "axis", [0, 1, 2])
        dataset = DatasetObject.create_dict_inputs("group", "dataset", [10, 20, 30])
        obj_h5dm.add(dataset, [axis])
        valid_filename = tempfile.mktemp(suffix=".h5")
        invalid_filename = tempfile.mktemp(suffix=".h5")
        try:
            obj_h5dm.save_hdf5_file(valid_filename)
            self.assertTrue(HDF5DataManager.validate_h5file_structure(valid_filename))
            
            # Create an invalid file
            with h5py.File(invalid_filename, 'w') as f:
                f.create_group("invalid_group")
            self.assertFalse(HDF5DataManager.validate_h5file_structure(invalid_filename))
        finally:
            os.remove(valid_filename)
            os.remove(invalid_filename)

    def test_load_invalid_file(self):
        invalid_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(invalid_filename, 'w') as f:
                f.create_group("invalid_group")
            with self.assertRaises(ValueError):
                HDF5DataManager.load_hdf5_file(invalid_filename)
        finally:
            os.remove(invalid_filename)

if __name__ == "__main__":
    unittest.main()
