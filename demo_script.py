import h5py
import numpy



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
