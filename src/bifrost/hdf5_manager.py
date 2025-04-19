## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import copy
import h5py
import json
import numpy

from bifrost import axes_manager
from bifrost import datasets_manager


## ###############################################################
## HDF5 DATA MANAGER
## ###############################################################
class HDF5DataManager:
  def __init__(self):
    self.dict_global_axes = {} # { axes_group: { axis_name: axes.AxisObject, ... }, ... }
    self.dict_datasets = {} # { dataset_group: { dataset_name: dataset.DatasetObject, ... }, ... }
    self.dict_axis_dependencies = {} # { (axis_group, axis_name): [ (dataset_group, dataset_name), ... ], ... }

  @staticmethod
  def validate_h5file_structure(file_path: str) -> bool:
    try:
      with h5py.File(file_path, "r") as file_pointer:
        if ("global_axes" not in file_pointer) or ("datasets" not in file_pointer): return False
        for axis_group in file_pointer["global_axes"].values():
          if not isinstance(axis_group, h5py.Group): return False
          for axis in axis_group.values():
            if ("units" not in axis.attrs) or ("notes" not in axis.attrs): return False
        for dataset_group in file_pointer["datasets"].values():
          if not isinstance(dataset_group, h5py.Group): return False
          for dataset in dataset_group.values():
            if ("values" not in dataset) or ("local_axes" not in dataset): return False
            if ("units" not in dataset.attrs) or ("notes" not in dataset.attrs): return False
        return True
    except Exception as exception:
      return False

  @classmethod
  def load_hdf5_file(cls, file_path):
    if not HDF5DataManager.validate_h5file_structure(file_path):
      raise ValueError(f"Invalid HDF5 file structure: {file_path}")
    obj_h5dm = cls() # initialise an instance of HDF5DataManager
    with h5py.File(file_path, "r") as h5_file:
      if "global_axes" in h5_file:
        for axis_group, h5_global_axes_group in h5_file["global_axes"].items():
          obj_h5dm.dict_global_axes[axis_group] = {}
          for axis_name, h5_global_axis in h5_global_axes_group.items():
            obj_h5dm.dict_global_axes[axis_group][axis_name] = axes_manager.AxisObject(
              group  = axis_group,
              name   = axis_name,
              values = numpy.array(h5_global_axis[:]),
              units  = h5_global_axis.attrs.get("units", ""),
              notes  = h5_global_axis.attrs.get("notes", ""),
            )
            axis_id = (axis_group, axis_name)
            str_axis_dependencies = h5_global_axis.attrs.get("dependencies", "[]")
            obj_h5dm.dict_axis_dependencies[axis_id] = [
              tuple(dependency.split("/"))
              for dependency in json.loads(str_axis_dependencies)
            ]
      if "datasets" in h5_file:
        for dataset_group, h5_datasets in h5_file["datasets"].items():
          obj_h5dm.dict_datasets[dataset_group] = {}
          for dataset_name, h5_dataset in h5_datasets.items():
            if not isinstance(h5_dataset, h5py.Group): continue
            dataset_values = numpy.array(h5_dataset["values"])
            dataset_units  = h5_dataset.attrs.get("units", "")
            dataset_notes  = h5_dataset.attrs.get("notes", "")
            list_axis_objs = []
            if "local_axes" in h5_dataset:
              h5_local_axes_group = h5_dataset["local_axes"]
              for axis_name in h5_local_axes_group:
                h5_local_axis = h5_local_axes_group[axis_name]
                axis_group    = h5_local_axis.attrs["group"]
                axis_name     = h5_local_axis.attrs["name"]
                axis_values   = numpy.array(h5_local_axis["values"])
                global_ref    = obj_h5dm.dict_global_axes.get(axis_group, {}).get(axis_name, None)
                axis_obj = axes_manager.AxisObject(
                  group  = axis_group,
                  name   = axis_name,
                  values = axis_values,
                  units  = "", # no units stored for local axes
                  notes  = "", # no notes stored for local axes
                  global_ref = global_ref, # set global reference
                )
                list_axis_objs.append(axis_obj)
            obj_h5dm.dict_datasets[dataset_group][dataset_name] = datasets_manager.DatasetObject(
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
      h5_global_axes = h5_file.create_group("global_axes")
      for axis_group, dict_axes_group in self.dict_global_axes.items():
        h5_global_axes_group = h5_global_axes.create_group(axis_group)
        for axis_name, obj_axis_global in dict_axes_group.items():
          ## store global axis values: super-set of all the values in the various instances of this axis
          h5_global_axis = h5_global_axes_group.create_dataset(axis_name, data=numpy.array(obj_axis_global.values))
          units = axes_manager.AxisObject.cast_units_to_string(obj_axis_global.units)
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
      h5_datasets = h5_file.create_group("datasets")
      for dataset_group, datasets in self.dict_datasets.items():
        h5_datasets_group = h5_datasets.create_group(dataset_group)
        for dataset_name, obj_dataset in datasets.items():
          h5_dataset = h5_datasets_group.create_group(dataset_name)
          h5_dataset.create_dataset("values", data=numpy.array(obj_dataset.values))
          units = datasets_manager.DatasetObject.cast_units_to_string(obj_dataset.units)
          h5_dataset.attrs["units"] = units
          h5_dataset.attrs["notes"] = obj_dataset.notes
          h5_local_axes = h5_dataset.create_group("local_axes")
          for axis_index, obj_axis_local in enumerate(obj_dataset.list_axis_objs):
            h5_local_axis = h5_local_axes.create_group(f"axis_{axis_index}")
            h5_local_axis.attrs["group"] = obj_axis_local.group
            h5_local_axis.attrs["name"] = obj_axis_local.name
            h5_local_axis.create_dataset("values", data=numpy.array(obj_axis_local.values))

  def add(self, dict_dataset, list_axis_dicts):
    dataset_group  = dict_dataset.get("group")
    dataset_name   = dict_dataset.get("name")
    dataset_id     = (dataset_group, dataset_name)
    dataset_values = dict_dataset.get("values")
    dataset_units  = dict_dataset.get("units")
    dataset_units  = datasets_manager.DatasetObject.cast_units_to_string(dataset_units)
    dataset_notes  = dict_dataset.get("notes")
    HDF5DataManager._validate_dimensions(dataset_values, list_axis_dicts)
    ## create the dataset group if it does not already exist
    if dataset_group not in self.dict_datasets: self.dict_datasets[dataset_group] = {}
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
      axis_units  = axes_manager.AxisObject.cast_units_to_string(axis_units)
      axis_notes  = dict_axis.get("notes")
      ## make sure that the manager remembers that the dataset has a dependency on the axis
      if axis_id    not in self.dict_axis_dependencies:          self.dict_axis_dependencies[axis_id] = []
      if dataset_id not in self.dict_axis_dependencies[axis_id]: self.dict_axis_dependencies[axis_id].append(dataset_id)
      ## store information relevant for initialising or updating (merging + reindexing) the dataset
      if bool_init_dataset:
        obj_axis_local = axes_manager.AxisObject(
          group  = axis_group,
          name   = axis_name,
          values = axis_values,
          units  = axis_units,
          notes  = axis_notes,
        )
        list_axis_objs.append(obj_axis_local)
      else: list_axis_values.append(axis_values)
      ## create the global version of the dataset group if it does not already exist
      if axis_group not in self.dict_global_axes: self.dict_global_axes[axis_group] = {}
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
        if axis_units != axes_manager.AxisUnits.NOT_SPECIFIED.value: obj_axis_global.units = axis_units
        if len(axis_notes) > 0: obj_axis_global.notes = axis_notes
    if bool_init_dataset:
      ## initialise the dataset object
      self.dict_datasets[dataset_group][dataset_name] = datasets_manager.DatasetObject(
        group  = dataset_group,
        name   = dataset_name,
        values = dataset_values,
        units  = dataset_units,
        notes  = dataset_notes,
        list_axis_objs = list_axis_objs,
      )
    ## update the dataset shape and values
    ## reindex using the input axis values to guide where `new` values should be inserted, or existing data should be overwritten
    else: self.dict_datasets[dataset_group][dataset_name].add(dataset_values_in=dataset_values, list_axis_values_in=list_axis_values)

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
      if not isinstance(units, (str, axes_manager.AxisUnits)):
        raise TypeError("Error: `new_units` was neither a string or an element from AxisUnits.")
      obj_axis_global.units = axes_manager.AxisObject.cast_units_to_string(units)
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
      new_axis_values = numpy.array([])
      if obj_axis_global is not None: new_axis_values = numpy.setdiff1d(obj_axis_global.values, obj_axis_local.values)
      list_new_axis_values_from_global.append(new_axis_values)
    ## only reindex if at least one local axis is missing values in the global axis.
    if any(
        axis_values.size > 0
        for axis_values in list_new_axis_values_from_global
      ): obj_dataset_copy.reindex(list_axis_values_in=list_new_axis_values_from_global)
    return obj_dataset_copy.get_dict()


## END OF MODULE