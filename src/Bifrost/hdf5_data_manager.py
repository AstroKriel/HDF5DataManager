from enum import Enum
import numpy as np
import h5py

class AxisType(Enum):
  TIME = "time"
  BIN_EDGES = "bin_edges"
  K_MODES = "k_modes"

class AxisUnits(Enum):
  NOT_SPECIFIED = "not_specified"
  DIMENSIONLESS = "dimensionless"
  T_TURB = "t_turb"
  K_TURB = "k_turb"

class DatasetUnits(Enum):
  NOT_SPECIFIED = "not_specified"
  DIMENSIONLESS = "dimensionless"

class HDF5DataManager:
  def __init__(self):
    self.dict_axes = {}  # {axis_group: {axis_name: {data, units, locked}}}
    self.dict_datasets = {}  # {dataset_group: {dataset_name: {data, units, axes}}}
    self.dict_metadata = {}
    self.set_axis_extended = set()
    self.set_axis_overwritten = set()
    self.set_axis_needs_reindexing = set()
    self.dict_lookup_axis_name2axis_group = {}  # {axis_name: axis_group}
    self.dict_lookup_axis_name2dataset_name = {}  # {axis_name: [dataset_name]}

  def add_axis(
      self,
      axis_group, axis_name, axis_values,
      axis_units = AxisUnits.NOT_SPECIFIED,
      bool_overwrite = False,
      bool_locked = False,
      bool_allow_non_monotonic = False
    ):
    """Adds an axis to the manager, with optional overwrite and locking."""
    status = {"success": False, "message": "", "needs_reindex": False}
    if not isinstance(axis_units, AxisUnits):
      status["message"] = f"Invalid axis unit: {axis_units}"
      return status
    if axis_group not in self.dict_axes:
      self.dict_axes[axis_group] = {}
    if axis_name in self.dict_axes[axis_group]:
      if self.dict_axes[axis_group][axis_name].get("locked", False):
        status["message"] = f"Error: Cannot modify '{axis_group}/{axis_name}' because it is locked."
        return status
      axis_values_old = self.dict_axes[axis_group][axis_name]["data"]
      if bool_overwrite:
        if not self._validate_axis(axis_values, status, bool_allow_non_monotonic):
          return status
        self.dict_axes[axis_group][axis_name] = {
          "data": axis_values,
          "units": axis_units.value,
          "locked": bool_locked
        }
        self.set_axis_overwritten.add(f"{axis_group}/{axis_name}")
        self.dict_lookup_axis_name2axis_group[axis_name] = axis_group  # Ensure lookup consistency
        status["needs_reindex"] = True
      else:
        axis_values_new = np.unique(np.concatenate((axis_values_old, axis_values)))
        if not self._validate_axis(axis_values_new, status, bool_allow_non_monotonic):
          return status
        self.dict_axes[axis_group][axis_name]["data"] = axis_values_new
        if len(axis_values_new) != len(axis_values_old):
          self.set_axis_extended.add(f"{axis_group}/{axis_name}")
          status["needs_reindex"] = True
    else:
      if not self._validate_axis(axis_values, status, bool_allow_non_monotonic):
        return status
      self.dict_axes[axis_group][axis_name] = {
        "data": axis_values,
        "units": axis_units.value,
        "locked": bool_locked
      }
      self.dict_lookup_axis_name2axis_group[axis_name] = axis_group
    status["success"] = True
    return status

  def _validate_axis(self, values, status, bool_allow_non_monotonic=False):
    """Checks axis validity: 1D, non-empty, monotonic (if required)."""
    if values.ndim != 1:
      status["message"] = "Axis must be a 1D array"
      return False
    if len(values) == 0:
      status["message"] = "Axis cannot be empty"
      return False
    if not bool_allow_non_monotonic and not np.all(np.diff(values) >= 0):
      status["message"] = "Axis values need to be monotonic"
      return False
    return True

  def add_data(
      self,
      dataset_group, dataset_name, dataset_values, list_axis_dicts,
      dataset_units = DatasetUnits.NOT_SPECIFIED,
      bool_overwrite = False,
      bool_locked = False
    ):
    """Adds a dataset and links it to its axes, with support for overwriting and locking."""
    status = {
      "success": False,
      "message": "",
      "needs_reindex": False
    }
    if not isinstance(dataset_units, DatasetUnits):
      status["message"] = f"Invalid dataset unit: {dataset_units}"
      return status
    if len(list_axis_dicts) != dataset_values.ndim:
      status["message"] = "Error: Dataset dimensions do not match the number of provided axes."
      return status
    list_missing_axes = [
      f"{dict_axis['type']}/{dict_axis['name']}"
      for dict_axis in list_axis_dicts
      if (dict_axis["type"] not in self.dict_axes) or (dict_axis["name"] not in self.dict_axes[dict_axis["type"]])
    ]
    if list_missing_axes:
      status["message"] = f"Error: The following axes are missing. Please add them first: {list_missing_axes}"
      return status
    if dataset_group not in self.dict_datasets:
      self.dict_datasets[dataset_group] = {}
    if dataset_name in self.dict_datasets[dataset_group]:
      existing_dataset = self.dict_datasets[dataset_group][dataset_name]
      if existing_dataset.get("locked", False):
        status["message"] = f"Error: Dataset `{dataset_group}/{dataset_name}` is locked and cannot be modified."
        return status
      if bool_overwrite:
        existing_dataset.update({
          "data": dataset_values,
          "units": dataset_units.value,
          "axes": [dict_axis["name"] for dict_axis in list_axis_dicts]
        })
        status["message"] = f"Warning: Dataset `{dataset_group}/{dataset_name}` was overwritten."
      else:
        status["message"] = f"Error: Dataset `{dataset_group}/{dataset_name}` already exists. Use overwrite=True to replace it."
        return status
    else:
      self.dict_datasets[dataset_group][dataset_name] = {
        "data": dataset_values,
        "units": dataset_units.value,
        "axes": [dict_axis["name"] for dict_axis in list_axis_dicts],
        "locked": bool_locked
      }
    for dict_axis in list_axis_dicts:
      axis_name = dict_axis["name"]
      self.dict_lookup_axis_name2dataset_name.setdefault(axis_name, []).append(dataset_name)
    status["needs_reindex"] = any(dict_axis["name"] in self.set_axis_needs_reindexing for dict_axis in list_axis_dicts)
    status["success"] = True
    return status


  def set_metadata(self, key, value):
    """Stores metadata in the manager."""
    self.dict_metadata[key] = value

  def save_to_hdf5(self, file_path):
    """Saves data to an HDF5 file."""
    with h5py.File(file_path, "w") as fp:
      self._save_axes(fp)
      self._save_datasets(fp)
      self._save_metadata(fp)

  def _save_axes(self, fp):
    """Writes axes to HDF5 under 'axes' group."""
    axis_group = fp.create_group("axes")
    for axis_group_name, axis_dict in self.dict_axes.items():
      group = axis_group.create_group(axis_group_name)
      for axis_name, axis_info in axis_dict.items():
        axis_group = group.create_group(axis_name)
        axis_group.create_dataset("data", data=axis_info["data"])
        axis_group.attrs["units"] = axis_info["units"]

  def _save_datasets(self, fp):
    """Writes datasets to HDF5 under 'datasets' group."""
    datasets_group = fp.create_group("datasets")
    for dataset_group_name, dataset_dict in self.dict_datasets.items():
      group = datasets_group.create_group(dataset_group_name)
      for dataset_name, dataset_info in dataset_dict.items():
        dataset_group = group.create_group(dataset_name)
        dataset_group.create_dataset("data", data=dataset_info["data"])
        dataset_group.attrs["units"] = dataset_info["units"]
        for axis_index, axis_name in enumerate(dataset_info["axes"]):
          axis_group_name = self.get_axis_group(axis_name) or "unknown_group"
          dataset_group.attrs[f"axis_{axis_index}_tag"] = f"/axes/{axis_group_name}/{axis_name}"

  def _save_metadata(self, fp):
    """Writes metadata to HDF5 under 'metadata' group."""
    metadata_group = fp.create_group("metadata")
    for key, value in self.dict_metadata.items():
      metadata_group.attrs[key] = value

  def get_axis_group(self, axis_name):
    """Retrieves the group of an axis (fast lookup)."""
    return self.dict_lookup_axis_name2axis_group.get(axis_name, None)


  # def reindex_data(self, axis_name):
  #   status = {
  #     "success": True,
  #     "message": "",
  #     "affected_datasets": []
  #   }
  #   try:
  #     axis_group = self.get_axis_group(axis_name)
  #     axis_len = len(self.dict_axes[axis_group][axis_name]["data"])
  #     for dg_name, dg in self.dict_datasets.items():
  #       for ds_name, ds in dg.items():
  #         if axis_name in ds["axes"]:
  #           idx = ds["axes"].index(axis_name)
  #           if ds["data"].shape[idx] != axis_len:
  #             status["affected_datasets"].append({
  #               "group": dg_name,
  #               "name": ds_name,
  #               "expected": axis_len,
  #               "actual": ds["data"].shape[idx]
  #             })
  #     if status["affected_datasets"]:
  #       status["message"] = "Reindexing required"
  #     else:
  #       status["message"] = "No reindexing needed"
  #     return status
  #   except Exception as e:
  #     status.update({"success": False, "message": str(e)})
  #     return status

  # def load_from_hdf5(self, file_path, bool_load_all_data : bool = False):
  #   with h5py.File(file_path, "r") as fp:
  #     self._load_axes(fp)
  #     self._load_datasets(fp, bool_load_all_data)
  #     self._load_metadata(fp)

  # def _load_datasets(self, file_pointer, load_all_data=False):
  #   datasets = file_pointer['datasets']
  #   if load_all_data:
  #     self.dict_datasets = { key: datasets[key][:] for key in datasets }  # Load all data if flag is set
  #   else: self.dict_datasets = { key: datasets[key] for key in datasets }  # Just store the references
  
  # def read_file_structure(self, file_path):
  #   with h5py.File(file_path, "r") as fp:
  #     self._load_axes(fp)
  #     self._load_dataset_structure(fp)
  #     self._load_metadata(fp)

  # def _load_axes(self, fp):
  #   self.dict_axes = {}
  #   for axis_group_name in fp["axes"]:
  #     group = fp[f"axes/{axis_group_name}"]
  #     self.dict_axes[axis_group_name] = {}
  #     for axis_name in group:
  #       axis_group = group[axis_name]
  #       self.dict_axes[axis_group_name][axis_name] = {
  #         "data": axis_group["data"][:],
  #         "units": axis_group.attrs["units"]
  #       }

  # def _load_dataset_structure(self, fp):
  #   self.dict_datasets = {}
  #   for dataset_group_name in fp["datasets"]:
  #     group = fp[f"datasets/{dataset_group_name}"]
  #     self.dict_datasets[dataset_group_name] = {}
  #     for dataset_name in group:
  #       dataset_group = group[dataset_name]
  #       self.dict_datasets[dataset_group_name][dataset_name] = {
  #         "units": dataset_group.attrs["units"],
  #         "axes": [],
  #         "shape": dataset_group["data"].shape
  #       }
  #       # Get all axis tags and sort them by index
  #       axis_tags = []
  #       for attr_name in dataset_group.attrs:
  #         if attr_name.startswith("axis_"):
  #           parts = attr_name.split("_")
  #           if len(parts) == 3 and parts[0] == "axis" and parts[2] == "tag":
  #             axis_index = int(parts[1])
  #             axis_tags.append((axis_index, dataset_group.attrs[attr_name]))
  #       # Sort by axis index and extract names
  #       axis_tags.sort(key=lambda x: x[0])
  #       self.dict_datasets[dataset_group_name][dataset_name]["axes"] = [
  #         tag.split("/")[-1] for _, tag in axis_tags
  #       ]

  # def load_specific_dataset(self, file_path, dataset_group, dataset_name):
  #   """Load a specific dataset from the HDF5 file."""
  #   with h5py.File(file_path, "r") as fp:
  #     if dataset_group not in fp["datasets"] or dataset_name not in fp[f"datasets/{dataset_group}"]:
  #       raise ValueError(f"Dataset {dataset_group}/{dataset_name} not found in the file.")
  #     dataset_group = fp[f"datasets/{dataset_group}"]
  #     dataset = dataset_group[dataset_name]
  #     if dataset_group not in self.dict_datasets:
  #       self.dict_datasets[dataset_group] = {}
  #     self.dict_datasets[dataset_group][dataset_name] = {
  #       "data": dataset["data"][:],
  #       "units": dataset.attrs["units"],
  #       "axes": [],
  #       "shape": dataset["data"].shape
  #     }
  #     # Get all axis tags and sort them by index
  #     axis_tags = []
  #     for attr_name in dataset.attrs:
  #       if attr_name.startswith("axis_"):
  #         parts = attr_name.split("_")
  #         if len(parts) == 3 and parts[0] == "axis" and parts[2] == "tag":
  #           axis_index = int(parts[1])
  #           axis_tags.append((axis_index, dataset.attrs[attr_name]))
  #     # Sort by axis index and extract names
  #     axis_tags.sort(key=lambda x: x[0])
  #     self.dict_datasets[dataset_group][dataset_name]["axes"] = [
  #       tag.split("/")[-1] for _, tag in axis_tags
  #     ]

  # def read_group(self, file_path, group_name):
  #   group_data = {}
  #   with h5py.File(file_path, "r") as fp:
  #     if group_name not in fp:
  #       raise ValueError(f"Group `{group_name}` not found in the file.")
  #     group = fp[group_name]
  #     for dataset_name, dataset in group.items():
  #       group_data[dataset_name] = {
  #         "data": dataset[:],
  #         "attrs": dict(dataset.attrs)
  #       }
  #       if "axes" in dataset.attrs:
  #         group_data[dataset_name]["axes"] = dataset.attrs["axes"]
  #   return group_data

  # def _load_metadata(self, fp):
  #   self.dict_metadata = {}
  #   metadata_group = fp["metadata"]
  #   for key, value in metadata_group.attrs.items():
  #     self.dict_metadata[key] = value

  # def cull_unused_axes(self):
  #   """
  #   Cull axes that are not referenced by any dataset.
  #   This should remove axes that are not used in any dataset.
  #   """
  #   used_axes = set()
  #   for dataset in self.dict_datasets.values():
  #     for axis_ref in dataset['axes']:
  #       used_axes.add(axis_ref['name'])
  #   axis_to_remove = [axis for axis in self.dict_axes if axis not in used_axes]
  #   for axis in axis_to_remove:
  #     del self.dict_axes[axis]
  #   return axis_to_remove

  # def check_consistency(self):
  #   axis_lengths = {}
  #   for dataset_group in self.dict_datasets.values():
  #     for dataset in dataset_group.values():
  #       for i, axis_name in enumerate(dataset["axes"]):
  #         # Get the actual dimension length from the data shape
  #         current_length = dataset["data"].shape[i]  # Remove len() here
  #         if axis_name not in axis_lengths:
  #           axis_lengths[axis_name] = current_length
  #         elif axis_lengths[axis_name] != current_length:
  #           raise ValueError(f"Inconsistent length for axis {axis_name}")

  # def get_axis(self, axis_name):
  #   for axis_group in self.dict_axes.values():
  #     if axis_name in axis_group:
  #       return axis_group[axis_name]
  #   raise ValueError(f"Axis `{axis_name}` not found")

  # def get_dataset(self, dataset_group, dataset_name):
  #   if (dataset_group in self.dict_datasets) and (dataset_name in self.dict_datasets[dataset_group]):
  #     return self.dict_datasets[dataset_group][dataset_name]
  #   raise ValueError(f"Dataset `{dataset_group}/{dataset_name}` not found")

  # def list_axes(self):
  #   return [
  #     (group, name)
  #     for group, axes in self.dict_axes.items()
  #     for name in axes
  #   ]

  # def list_datasets(self):
  #   return [
  #     (group, name)
  #     for group, datasets in self.dict_datasets.items()
  #     for name in datasets
  #   ]

  # def remove_dataset(self, dataset_group, dataset_name):
  #   if (dataset_group in self.dict_datasets) and (dataset_name in self.dict_datasets[dataset_group]):
  #     del self.dict_datasets[dataset_group][dataset_name]
  #     if not self.dict_datasets[dataset_group]:
  #       del self.dict_datasets[dataset_group]
  #     return True
  #   else: return False

  # def remove_axis(self, axis_group, axis_name):
  #   if axis_group in self.dict_axes and axis_name in self.dict_axes[axis_group]:
  #     del self.dict_axes[axis_group][axis_name]
  #     if not self.dict_axes[axis_group]:
  #       del self.dict_axes[axis_group]
  #     return True
  #   else: return False