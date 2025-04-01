import os
import h5py
import numpy as np
import tempfile
import unittest
from bifrost import hdf5_data_manager as h5dm

class TestHDF5DataManager(unittest.TestCase):

    def setUp(self):
        self.obj_h5dm = h5dm.HDF5DataManager()

    def test_invalid_axis_creation(self):
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "valid_name", "invalid_values")  # no axis values
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "valid_name", [[1, 2], [3, 4]])  # axis values are not a flat set of values
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "valid_name", [2, 1, 4, 3])  # axis values are not monotonically increasing
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "valid_name", [1, 1, 2, 2])  # axis values are not unique
        with self.assertRaises(TypeError):
            h5dm.AxisObject.create_dict_inputs("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_invalid_dataset_creation(self):
        with self.assertRaises(ValueError):
            h5dm.DatasetObject.create_dict_inputs("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            h5dm.DatasetObject.create_dict_inputs("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            h5dm.DatasetObject.create_dict_inputs("valid_group", "valid_name", "invalid_values")  # no dataset values
        with self.assertRaises(TypeError):
            h5dm.DatasetObject.create_dict_inputs("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

    def test_axis_creation(self):
        axis_values = np.arange(1000)
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED, notes="Test axis")
        self.assertEqual(dict_axis["group"], "axis_group")
        self.assertEqual(dict_axis["name"], "axis_name")
        np.testing.assert_array_equal(dict_axis["values"], axis_values)
        self.assertEqual(dict_axis["units"], h5dm.AxisUnits.NOT_SPECIFIED.value)
        self.assertEqual(dict_axis["notes"], "Test axis")

    def test_dataset_creation(self):
        dataset_values = np.random.rand(1000) * 100
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.DIMENSIONLESS, notes="Test dataset")
        self.assertEqual(dict_dataset["group"], "dataset_group")
        self.assertEqual(dict_dataset["name"], "dataset_name")
        np.testing.assert_array_equal(dict_dataset["values"], dataset_values)
        self.assertEqual(dict_dataset["units"], h5dm.DatasetUnits.DIMENSIONLESS.value)
        self.assertEqual(dict_dataset["notes"], "Test dataset")

    def test_add_data(self):
        length = 100
        axis_values = np.arange(length)
        dataset_values = np.random.rand(length)
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
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
        axis_dict_1 = h5dm.AxisObject.create_dict_inputs("axis_group_1", "axis_name_1", axis_values_1, h5dm.AxisUnits.NOT_SPECIFIED)
        axis_dict_2 = h5dm.AxisObject.create_dict_inputs("axis_group_2", "axis_name_2", axis_values_2, h5dm.AxisUnits.NOT_SPECIFIED)
        dataset_dict_1 = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_1", dataset_values_1, h5dm.DatasetUnits.NOT_SPECIFIED)
        dataset_dict_2 = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_2", dataset_values_2, h5dm.DatasetUnits.NOT_SPECIFIED)
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
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", new_axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", new_dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(new_dataset_dict, [new_axis_dict])
        global_data = self.obj_h5dm.get_global_dataset("dataset_group", "dataset_name")
        self.assertIsNotNone(global_data)
        expected_values = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(global_data["values"], expected_values)

    def test_reindexing_dataset_to_get_values_with_global_axis_values(self):
        axis_values = np.array([0, 1, 2])
        dataset_values = np.array([10, 20, 30])
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group_1", "dataset_name_1", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        new_dataset_values = np.array([40, 50])
        new_axis_values = np.array([3, 4])
        new_axis_dict = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", new_axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        new_dataset_dict = h5dm.DatasetObject.create_dict_inputs("dataset_group_2", "dataset_name_2", new_dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        self.obj_h5dm.add(new_dataset_dict, [new_axis_dict])
        global_data = self.obj_h5dm.get_global_dataset("dataset_group_1", "dataset_name_1")
        self.assertIsNotNone(global_data)
        expected_values = np.array([10, 20, 30, np.nan, np.nan])
        np.testing.assert_array_equal(global_data["values"], expected_values)

    def test_dataset_reindexing_expands_correctly(self):
        axis_values = np.array([3, 4])
        dataset_values = np.array([30, 40])
        axis = h5dm.AxisObject("group1", "axis1", axis_values)
        dataset = h5dm.DatasetObject("group1", "dataset1", dataset_values, [axis])
        axis_values_global = np.array([0, 1, 2, 3, 4, 5, 6])
        dataset.reindex([axis_values_global])
        self.assertEqual(dataset.values.shape[0], len(axis_values_global))
        np.testing.assert_array_equal(dataset.values, [np.nan, np.nan, np.nan, 30, 40, np.nan, np.nan])

    def test_dataset_overwrite_with_new_data(self):
        axis_values = np.array([1, 2])
        dataset_values = np.array([10, 20])
        axis = h5dm.AxisObject("group1", "axis1", axis_values)
        dataset = h5dm.DatasetObject("group1", "dataset1", dataset_values, [axis])
        new_axis_values = np.array([0, 1, 2, 3])
        new_dataset_values = np.array([0, 11, 22, 30])
        dataset.reindex([new_axis_values], new_dataset_values)
        expected_values = np.array([0, 11, 22, 30])
        np.testing.assert_array_equal(dataset.values, expected_values)

    def test_add_and_update_metadata(self):
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2])
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30])
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        # Update metadata
        self.obj_h5dm.update_global_axis_metadata("axis_group", "axis_name", units=h5dm.AxisUnits.T_TURB, notes="Updated notes")
        updated_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["units"], h5dm.AxisUnits.T_TURB.value)
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["notes"], "Updated notes")

    def test_extend_existing_dataset(self):
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2])
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30])
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        # Extend dataset
        new_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [3, 4])
        new_data = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [40, 50])
        self.obj_h5dm.add(new_data, [new_axis])
        extended_dataset = self.obj_h5dm.get_global_dataset("dataset_group", "dataset_name")
        np.testing.assert_array_equal(extended_dataset["values"], [10, 20, 30, 40, 50])

    # Edge case 1: Empty Axis Values
    def test_empty_axis_values(self):
        empty_axis_values = []
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", empty_axis_values, h5dm.AxisUnits.NOT_SPECIFIED)

    # Edge case 3: Adding dataset with mismatched axis shapes
    def test_mismatched_axis_and_dataset_shapes(self):
        axis_values = np.arange(3)
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        dataset_values = np.random.rand(4)  # Dataset shape does not match axis shape
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [dict_axis])

    # Edge case 4: Axis values with NaN values
    def test_axis_values_with_nan(self):
        axis_values = np.array([1, 2, np.nan, 4])
        with self.assertRaises(ValueError):
            h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)

    # Edge case 5: Invalid axis unit type
    def test_invalid_axis_unit_type(self):
        axis_values = np.arange(3)
        with self.assertRaises(TypeError):
            h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, "invalid_unit")

    # Edge case 6: Invalid dataset unit type
    def test_invalid_dataset_unit_type(self):
        dataset_values = np.random.rand(3)
        with self.assertRaises(TypeError):
            h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, "invalid_unit")

    # Edge case 7: Adding new axis values that overlap
    def test_adding_overlap_axis_values(self):
        axis_values_1 = np.array([1, 2, 3])
        axis_values_2 = np.array([2, 3, 4])
        dataset_values = np.random.rand(3)
        axis_dict_1 = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values_1, h5dm.AxisUnits.NOT_SPECIFIED)
        axis_dict_2 = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values_2, h5dm.AxisUnits.NOT_SPECIFIED)
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
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
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
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
        dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, h5dm.DatasetUnits.NOT_SPECIFIED)
        # Add the dataset with a missing axis
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [])

    # Edge case 11: Adding datasets with different units
    def test_adding_datasets_with_different_units(self):
        axis_values = np.arange(3)
        dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, h5dm.AxisUnits.NOT_SPECIFIED)
        # Dataset with dimensionless units
        dataset_values_1 = np.random.rand(3)
        dataset_dict_1 = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_1", dataset_values_1, h5dm.DatasetUnits.DIMENSIONLESS)
        # Dataset with different units
        dataset_values_2 = np.random.rand(3)
        dataset_dict_2 = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_2", dataset_values_2, h5dm.DatasetUnits.NOT_SPECIFIED)
        # Adding datasets with different units
        self.obj_h5dm.add(dataset_dict_1, [dict_axis])
        self.obj_h5dm.add(dataset_dict_2, [dict_axis])
        dict_dataset_1 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_1")
        dict_dataset_2 = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_2")
        self.assertIsNotNone(dict_dataset_1)
        self.assertIsNotNone(dict_dataset_2)
    
    def test_global_axis_inheritance(self):
        # Update the global axis metadata using the update function
        self.dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=h5dm.AxisUnits.DIMENSIONLESS, notes="original axis")
        self.dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=h5dm.DatasetUnits.DIMENSIONLESS, notes="test dataset")
        self.obj_h5dm.add(self.dict_dataset, [self.dict_axis])
        new_units = h5dm.AxisUnits.T_TURB
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
            dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=h5dm.AxisUnits.T_TURB, notes="original axis")
            dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=h5dm.DatasetUnits.DIMENSIONLESS, notes="test dataset")
            obj_h5dm_save = h5dm.HDF5DataManager()
            obj_h5dm_save.add(dict_dataset, [dict_axis])
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            self.assertTrue(h5dm.HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_invalid_file_structure(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group("invalid_group")
            self.assertFalse(h5dm.HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_nonexistent_file(self):
        non_existent_file = "non_existent_file.h5"
        self.assertFalse(h5dm.HDF5DataManager.validate_h5file_structure(non_existent_file))

    def test_missing_required_groups(self):
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group("global_axes")  # Missing "datasets" group
            self.assertFalse(h5dm.HDF5DataManager.validate_h5file_structure(tmp_filename))
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
            self.assertFalse(h5dm.HDF5DataManager.validate_h5file_structure(tmp_filename))
        finally:
            os.remove(tmp_filename)

    def test_save_and_load_hdf5_file(self):
        # Save the current manager to a temporary HDF5 file
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            dict_axis = h5dm.AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2], units=h5dm.AxisUnits.T_TURB, notes="original axis")
            dict_dataset = h5dm.DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30], units=h5dm.DatasetUnits.DIMENSIONLESS, notes="test dataset")
            obj_h5dm_save = h5dm.HDF5DataManager()
            obj_h5dm_save.add(dict_dataset, [dict_axis])
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            obj_h5dm_load = h5dm.HDF5DataManager.load_hdf5_file(tmp_filename)
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
        axis1 = h5dm.AxisObject.create_dict_inputs("group1", "axis1", [0, 1, 2])
        axis2 = h5dm.AxisObject.create_dict_inputs("group2", "axis2", [3, 4, 5])
        dataset1 = h5dm.DatasetObject.create_dict_inputs("group1", "dataset1", [10, 20, 30])
        dataset2 = h5dm.DatasetObject.create_dict_inputs("group2", "dataset2", [40, 50, 60])
        obj_h5dm_save = h5dm.HDF5DataManager()
        obj_h5dm_save.add(dataset1, [axis1])
        obj_h5dm_save.add(dataset2, [axis2])
        tmp_filename = tempfile.mktemp(suffix=".h5")
        try:
            obj_h5dm_save.save_hdf5_file(tmp_filename)
            obj_h5dm_load = h5dm.HDF5DataManager.load_hdf5_file(tmp_filename)
            loaded_dataset1 = obj_h5dm_load.get_local_dataset("group1", "dataset1")
            loaded_dataset2 = obj_h5dm_load.get_local_dataset("group2", "dataset2")
            np.testing.assert_array_equal(loaded_dataset1["values"], [10, 20, 30])
            np.testing.assert_array_equal(loaded_dataset2["values"], [40, 50, 60])
        finally:
            os.remove(tmp_filename)

    def test_file_structure_validation(self):
        # Create a valid file
        obj_h5dm = h5dm.HDF5DataManager()
        axis = h5dm.AxisObject.create_dict_inputs("group", "axis", [0, 1, 2])
        dataset = h5dm.DatasetObject.create_dict_inputs("group", "dataset", [10, 20, 30])
        obj_h5dm.add(dataset, [axis])
        valid_filename = tempfile.mktemp(suffix=".h5")
        invalid_filename = tempfile.mktemp(suffix=".h5")
        try:
            obj_h5dm.save_hdf5_file(valid_filename)
            self.assertTrue(h5dm.HDF5DataManager.validate_h5file_structure(valid_filename))
            
            # Create an invalid file
            with h5py.File(invalid_filename, 'w') as f:
                f.create_group("invalid_group")
            self.assertFalse(h5dm.HDF5DataManager.validate_h5file_structure(invalid_filename))
        finally:
            os.remove(valid_filename)
            os.remove(invalid_filename)

    def test_load_invalid_file(self):
        invalid_filename = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(invalid_filename, 'w') as f:
                f.create_group("invalid_group")
            with self.assertRaises(ValueError):
                h5dm.HDF5DataManager.load_hdf5_file(invalid_filename)
        finally:
            os.remove(invalid_filename)

if __name__ == "__main__":
    unittest.main()
