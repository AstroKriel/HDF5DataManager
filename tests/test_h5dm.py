import os
import unittest
import numpy as np
from Bifrost.hdf5_data_manager import HDF5DataManager, AxisUnits, DatasetUnits

class TestHDF5DataManager(unittest.TestCase):

  def setUp(self):
    """Setup the test environment for each test."""
    self.h5dmanager = HDF5DataManager()

  def test_add_axis_valid(self):
    """Test adding a valid axis."""
    axes_values = np.array([1, 2, 3])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    self.assertTrue(h5dm_status["success"])
    self.assertEqual(
      self.h5dmanager.dict_axes["time"]["sim_time"]["data"].tolist(),
      axes_values.tolist()
    )

  def test_add_axis_invalid_units(self):
    """Test adding an axis with invalid units."""
    axes_values = np.array([1, 2, 3])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = "invalid_unit" # raw string instead of one of the standardised Enum types (users should add their units there)
    )
    self.assertFalse(h5dm_status["success"])
    self.assertEqual(h5dm_status["message"], "Invalid axis unit: invalid_unit")

  def test_add_locked_axis(self):
    """Test that modifying a locked axis fails."""
    axes_values = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB,
      bool_locked = True
    )
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = np.array([4, 5, 6]),
      axis_units  = AxisUnits.T_TURB
    )
    self.assertFalse(h5dm_status["success"])
    self.assertEqual(h5dm_status["message"], "Error: Cannot modify 'time/sim_time' because it is locked.")

  def test_add_axis_overwrite(self):
    """Test overwriting an existing axis."""
    axes_values_1 = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values_1,
      axis_units  = AxisUnits.T_TURB
    )
    axes_values_2 = np.array([4, 5, 6])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group     = "time",
      axis_name      = "sim_time",
      axis_values    = axes_values_2,
      axis_units     = AxisUnits.T_TURB,
      bool_overwrite = True
    )
    self.assertTrue(h5dm_status["success"])
    self.assertEqual(
      self.h5dmanager.dict_axes["time"]["sim_time"]["data"].tolist(),
      axes_values_2.tolist()
    )

  def test_add_axis_extend(self):
    """Test extending an existing axis."""
    axes_values_1 = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values_1,
      axis_units  = AxisUnits.T_TURB
    )
    axes_values_2 = np.array([4, 5])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values_2,
      axis_units  = AxisUnits.T_TURB
    )
    self.assertTrue(h5dm_status["success"])
    self.assertEqual(
      self.h5dmanager.dict_axes["time"]["sim_time"]["data"].tolist(),
      np.concatenate([axes_values_1, axes_values_2]).tolist()
    )

  def test_add_data_valid(self):
    """Test adding a valid dataset."""
    axes_values = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    dataset_values = np.array([1.0, 2.0, 3.0])
    h5dm_status = self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "dataset1",
      dataset_values  = dataset_values,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "sim_time"}]
    )
    self.assertTrue(h5dm_status["success"])
    self.assertIn("group1", self.h5dmanager.dict_datasets)
    self.assertIn("dataset1", self.h5dmanager.dict_datasets["group1"])

  def test_add_data_missing_axes(self):
    """Test adding a dataset with missing axes."""
    dataset_values = np.array([1.0, 2.0, 3.0])
    h5dm_status = self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "dataset1",
      dataset_values  = dataset_values,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "missing_axis"}]
    )
    self.assertFalse(h5dm_status["success"])
    self.assertEqual(h5dm_status["message"], "Error: The following axes are missing. Please add them first: ['time/missing_axis']")

  def test_add_data_overwrite(self):
    """Test overwriting an existing dataset."""
    axes_values = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    dataset_values = np.array([1.0, 2.0, 3.0])
    self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "dataset1",
      dataset_values  = dataset_values,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "sim_time"}]
    )
    new_dataset_values = np.array([4.0, 5.0, 6.0])
    h5dm_status = self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "dataset1",
      dataset_values  = new_dataset_values,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "sim_time"}],
      bool_overwrite  = True
    )
    self.assertTrue(h5dm_status["success"])
    self.assertEqual(
      self.h5dmanager.dict_datasets["group1"]["dataset1"]["data"].tolist(),
      new_dataset_values.tolist()
    )

  def test_save_to_hdf5(self):
    """Test saving to HDF5."""
    axes_values = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    dataset_values = np.array([1.0, 2.0, 3.0])
    self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "dataset1",
      dataset_values  = dataset_values,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "sim_time"}]
    )
    file_path = "test.h5"
    self.h5dmanager.save_to_hdf5(file_path=file_path)
    self.assertTrue(os.path.exists(file_path))

  def test_empty_axis(self):
    """Test adding an empty axis."""
    axes_values = np.array([])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    self.assertFalse(h5dm_status["success"])
    self.assertEqual(h5dm_status["message"], "Axis cannot be empty")

  def test_non_monotonic_axis(self):
    """Test adding a non-monotonic axis."""
    axes_values = np.array([1, 3, 2])
    h5dm_status = self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    self.assertFalse(h5dm_status["success"])
    self.assertEqual(h5dm_status["message"], "Axis values need to be monotonic")

  def test_large_dataset(self):
    """Test adding a very large dataset."""
    axes_values = np.array([1, 2, 3])
    self.h5dmanager.add_axis(
      axis_group  = "time",
      axis_name   = "sim_time",
      axis_values = axes_values,
      axis_units  = AxisUnits.T_TURB
    )
    large_dataset = np.random.rand(1000000)
    h5dm_status = self.h5dmanager.add_data(
      dataset_group   = "group1",
      dataset_name    = "large_dataset",
      dataset_values  = large_dataset,
      dataset_units   = DatasetUnits.DIMENSIONLESS,
      list_axis_dicts = [{"type": "time", "name": "sim_time"}]
    )
    self.assertTrue(h5dm_status["success"])

if __name__ == '__main__':
  unittest.main()
