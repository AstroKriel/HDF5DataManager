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
        ## the following is equivelant to np.array(sorted(set(self.values) | set(values)))
        self.values = np.unique(np.concatenate((self.values, np.array(values))))

    def __eq__(self, obj_axis):
        if isinstance(obj_axis, AxisObject):
            bool_properties_are_the_same = all([
                self.group == obj_axis.group,
                self.name  == obj_axis.name,
                np.array_equal(self.values, obj_axis.values),
                self.units == obj_axis.units,
                self.notes == obj_axis.notes,
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

    def add(self, dataset_values, list_axis_values):
        self.reindex(
            dataset_values   = dataset_values,
            list_axis_values = list_axis_values,
        )

    def reindex(self, list_axis_values_in, dataset_values_in=None):
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
        ## if no `in` data was provided, then the new space will be filled with nans
        updated_dataset_shape = tuple(
            len(obj_axis_updated.values)
            for obj_axis_updated in self.list_axis_objs # this has now been updated
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
            ## `in` may not necessarily be `new`
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
        self.dict_axes_global = {}  # {group: {name: AxisObject}}
        self.dict_datasets = {}  # {group: {name: DatasetObject}}
        self.dict_axis_dependencies = {}  # { (axis_group, axis_name): [(dataset_group, dataset_name), ...] }

    def add(self, dict_dataset, list_axis_dicts):
        ## probs add a flag to overwrite axis & dataset properties
        ## there needs to be alignment between dataset and axis:
        ## 1. dict_dataset["values"].ndim == len(list_axis_dicts)
        ## 2. dict_dataset["values"].shape[i] == len(list_axis_dicts[i]["values"])
        ## the following properties should be gauranteed, at least with default values defined by the class that creates the dict
        dataset_group  = dict_dataset.get("group")
        dataset_name   = dict_dataset.get("name")
        dataset_id     = (dataset_group, dataset_name)
        dataset_values = dict_dataset.get("values")
        dataset_units  = dict_dataset.get("units")
        dataset_notes  = dict_dataset.get("notes")
        ## create the dataset group if it does not exist
        if dataset_group not in self.dict_datasets:
            self.dict_datasets[dataset_group] = {}
        ## check if the dataset will need to be created
        bool_init_dataset = dataset_name not in self.dict_datasets[dataset_group]
        ## if so, then axis objects will need to be stored to initialise the dataset
        list_axis_objs_in = []
        ## otherwise, only the input axis values need to be stored to inform updating and reindexing the dataset
        list_axis_values_in = []
        for dict_axis in list_axis_dicts:
            axis_group  = dict_axis.get("group")
            axis_name   = dict_axis.get("name")
            axis_id     = (axis_group, axis_name)
            axis_values = dict_axis.get("values")
            axis_units  = dict_axis.get("units")
            axis_notes  = dict_axis.get("notes")
            ## make sure the manager acknowledges that the dataset has a dependency on the axis
            if axis_id not in self.dict_axis_dependencies:
                self.dict_axis_dependencies[axis_id] = []
            if dataset_id not in self.dict_axis_dependencies[axis_id]:
                self.dict_axis_dependencies[axis_id].append(dataset_id)
            ## store information relevant for initialising/updating the dataset
            if bool_init_dataset:
                obj_axis_in = AxisObject(
                    group  = axis_group,
                    name   = axis_name,
                    values = axis_values,
                    units  = axis_units,
                    notes  = axis_notes,
                )
                list_axis_objs_in.append(obj_axis_in)
            else: list_axis_values_in.append(axis_values)
            ## create the global axis group if it does not exist
            if axis_group not in self.dict_axes_global:
                self.dict_axes_global[axis_group] = {}
            ## initialise the global axis object if it does not exist
            if axis_name not in self.dict_axes_global[axis_group]:
                ## use a copy of the local axis object
                ## a deep-copy is necessary, so that the global values can be extended without affecting the local dataset axis
                ## todo: think about whether/how the units/notes should be maintained (the same axis group/name may not have the same properties?)
                self.dict_axes_global[axis_group][axis_name] = copy.deepcopy(obj_axis_in)
            ## make sure that the global axis object conatins the superset of all the axis instances
            else: self.dict_axes_global[axis_group][axis_name].add(axis_values)
        if bool_init_dataset:
            ## initialise the dataset object
            self.dict_datasets[dataset_group][dataset_name] = DatasetObject(
                group  = dataset_group,
                name   = dataset_name,
                values = dataset_values,
                units  = dataset_units,
                notes  = dataset_notes,
                list_axis_objs = list_axis_objs_in,
            )
        else:
            ## update the dataset shape and values; reindex using the input axis values to guide where `new` values should be inserted
            self.dict_datasets[dataset_group][dataset_name].add(
                dataset_values   = dataset_values,
                list_axis_values = list_axis_values_in,
            )

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

if __name__ == '__main__':
    unittest.main()
