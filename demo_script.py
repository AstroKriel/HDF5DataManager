import h5py
import numpy

class HDF5DataManager:
  def __init__(self, filepath):
    self.filepath = filepath

  def create_axis(self, axis_name, axis_data):
    axis_type = axis_name.split("_")[0]
    with h5py.File(self.filename, "a") as hdf_file:
      axes_group = hdf_file.require_group("axes")
      axis_subgroup = axes_group.create_group(axis_name)
      axis_subgroup.create_dataset("data", data=axis_data)
      axis_subgroup.attrs["axis_type"] = axis_type

  def create_dataset(self, dataset_name, data, axes=None, axis_tags=None):
    with h5py.File(self.filepath, "a") as hdf_file:
      dataset_group = hdf_file.require_group("datasets")
      dataset_subgroup = dataset_group.create_group(dataset_name)
      dataset_subgroup.create_dataset("data", data=data)
      if axes:
        axes_group = dataset_subgroup.create_group("axes")
        for axis_name, axis_data in axes.items():
          axes_group.create_dataset(axis_name, data=axis_data)
      if axis_tags:
        for axis_name, axis_tag in axis_tags.items():
          dataset_subgroup.attrs[f"{axis_name}_axis_tag"] = axis_tag

  def reindex_axis(self, axis_name, axis_type, new_axis_values):
    with h5py.File(self.filepath, "a") as hdf_file:
      axes_group = hdf_file["axes"]
      axis_group = axes_group[axis_name]
      existing_axis_values = axis_group["data"][:]
      # Combine existing and new axis values
      all_axis_values = numpy.concatenate((existing_axis_values, new_axis_values))
      all_axis_values = numpy.sort(all_axis_values)
      # Update axis
      axis_group["data"][:] = all_axis_values
      # Reindex associated datasets
      self.reindex_datasets(axis_name, axis_type, all_axis_values)

  def reindex_datasets(self, axis_name, axis_type, new_axis_values):
    with h5py.File(self.filepath, "a") as hdf_file:
      datasets_group = hdf_file["datasets"]
      for dataset_name in datasets_group:
        dataset_group = datasets_group[dataset_name]
        if f"{axis_type}_axis_tag" in dataset_group.attrs and dataset_group.attrs[f"{axis_type}_axis_tag"] == f"/axes/{axis_name}":
          existing_data = dataset_group["data"][:]
          existing_axis_values = new_axis_values[:existing_data.shape[0]]
          # Create new array with NaNs where measurements are missing
          if axis_type == "time":
            new_data = numpy.full((len(new_axis_values), existing_data.shape[1]), numpy.nan)
          elif axis_type == "frequency":
            new_data = numpy.full((existing_data.shape[0], len(new_axis_values)), numpy.nan)
          else: raise ValueError("Unsupported axis type")
          new_data[:existing_data.shape[0], :] = existing_data
          # Update dataset
          dataset_group["data"][:] = new_data

  def read_dataset(self, dataset_name):
    with h5py.File(self.filepath, "r") as hdf_file:
      dataset_group = hdf_file["datasets"][dataset_name]
      data = dataset_group["data"][:]
      axes = {}
      if "axes" in dataset_group:
        axes_group = dataset_group["axes"]
        for axis_name in axes_group:
          axes[axis_name] = axes_group[axis_name][:]
      return data, axes

def main():
  filepath = "example.h5"
  manager = HDF5DataManager(filepath)
  # Create axes
  time_values = numpy.array([1, 2, 3])
  manager.create_axis("time_axis", time_values)
  frequency_values = numpy.array([4, 5, 6])
  manager.create_axis("frequency_axis", frequency_values)
  # Create datasets
  spectrum_data = numpy.array([[10, 20], [30, 40], [50, 60]])
  manager.create_dataset("spectrum", spectrum_data, axis_tags={"time_axis": "/axes/time_axis", "frequency_axis": "/axes/frequency_axis"})
  # Reindex time axis
  new_time_values = numpy.array([0, 4])  # New time values
  manager.reindex_axis("time_axis", "time", new_time_values)
  # Read and verify dataset
  data, axes = manager.read_dataset("spectrum")
  print(data)
  print(axes)
  # Test assertions
  assert data.shape == (5, 2)  # Expected shape after reindexing
  assert numpy.isnan(data[0, 0])  # First row should be NaN due to reindexing
  assert numpy.isnan(data[4, 0])  # Last row should be NaN due to reindexing
  assert data[1, 0] == 10  # Original data should remain unchanged

if __name__ == "__main__":
  main()

## end of demo script
