import unittest
import numpy as np
import tempfile
import os
import h5py
from your_module import HDF5DataManager, AxisObject, DatasetObject, AxisUnits, DatasetUnits

class TestAxisObjectCreation(unittest.TestCase):

    def setUp(self):
        self.obj_h5dm = HDF5DataManager()

    def test_valid_axis_creation(self):
        axis_values = np.arange(1000)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED, notes="Test axis")
        self.assertEqual(dict_axis["group"], "axis_group")
        self.assertEqual(dict_axis["name"], "axis_name")
        np.testing.assert_array_equal(dict_axis["values"], axis_values)
        self.assertEqual(dict_axis["units"], AxisUnits.NOT_SPECIFIED.value)
        self.assertEqual(dict_axis["notes"], "Test axis")
    
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

class TestDatasetObjectCreation(unittest.TestCase):

    def setUp(self):
        self.obj_h5dm = HDF5DataManager()

    def test_valid_dataset_creation(self):
        dataset_values = np.random.rand(1000) * 100
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.DIMENSIONLESS, notes="Test dataset")
        self.assertEqual(dict_dataset["group"], "dataset_group")
        self.assertEqual(dict_dataset["name"], "dataset_name")
        np.testing.assert_array_equal(dict_dataset["values"], dataset_values)
        self.assertEqual(dict_dataset["units"], DatasetUnits.DIMENSIONLESS.value)
        self.assertEqual(dict_dataset["notes"], "Test dataset")
    
    def test_invalid_dataset_creation(self):
        with self.assertRaises(ValueError):
            DatasetObject.create_dict_inputs("", "valid_name", [1, 2, 3])  # empty group
        with self.assertRaises(ValueError):
            DatasetObject.create_dict_inputs("valid_group", "", [1, 2, 3])  # empty name
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("valid_group", "valid_name", "invalid_values")  # no dataset values
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("valid_group", "valid_name", [1, 2, 3], "invalid_units")  # invalid units

class TestHDF5DataManagerCore(unittest.TestCase):
    def setUp(self):
        self.obj_h5dm = HDF5DataManager()
    
    def test_create_and_get_local_dataset(self):
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

    def test_add_multiple_datasets_with_a_shared_axis(self):
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

    def test_get_nonexistent_datasets(self):
        self.assertIsNone(self.obj_h5dm.get_local_dataset("nonexistent_group", "nonexistent_name"))
        self.assertIsNone(self.obj_h5dm.get_global_dataset("nonexistent_group", "nonexistent_name"))

class TestHDF5DataManagerExtending(unittest.TestCase):
    def setUp(self):
        self.obj_h5dm = HDF5DataManager()
    
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

class TestHDF5DataManagerMetadata(unittest.TestCase):
    def setUp(self):
        self.obj_h5dm = HDF5DataManager()
    
    def test_add_and_update_metadata(self):
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", [0, 1, 2])
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", [10, 20, 30])
        self.obj_h5dm.add(dict_dataset, [dict_axis])
        # Update metadata
        self.obj_h5dm.update_global_axis_metadata("axis_group", "axis_name", units=AxisUnits.T_TURB, notes="Updated notes")
        updated_dataset = self.obj_h5dm.get_local_dataset("dataset_group", "dataset_name")
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["units"], AxisUnits.T_TURB.value)
        self.assertEqual(updated_dataset["list_axis_dicts"][0]["notes"], "Updated notes")
    
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

class TestHDF5DataManagerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.obj_h5dm = HDF5DataManager()
    
    def test_empty_axis_values(self):
        empty_axis_values = []
        with self.assertRaises(ValueError):
            AxisObject.create_dict_inputs("axis_group", "axis_name", empty_axis_values, AxisUnits.NOT_SPECIFIED)
    
    def test_invalid_axis_unit_type(self):
        axis_values = np.arange(3)
        with self.assertRaises(TypeError):
            AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, "invalid_unit")
        
    def test_invalid_dataset_unit_type(self):
        dataset_values = np.random.rand(3)
        with self.assertRaises(TypeError):
            DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, "invalid_unit")
    
    def test_mismatched_axis_and_dataset_shapes(self):
        axis_values = np.arange(3)
        dict_axis = AxisObject.create_dict_inputs("axis_group", "axis_name", axis_values, AxisUnits.NOT_SPECIFIED)
        dataset_values = np.random.rand(4)  # Dataset shape does not match axis shape
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [dict_axis])
    
    def test_missing_axis_values(self):
        dataset_values = np.random.rand(3)  # Dataset does not have missing values in this case
        dict_dataset = DatasetObject.create_dict_inputs("dataset_group", "dataset_name", dataset_values, DatasetUnits.NOT_SPECIFIED)
        # Add the dataset with a missing axis
        with self.assertRaises(ValueError):
            self.obj_h5dm.add(dict_dataset, [])
    
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

class TestHDF5DataManagerSaveLoad(unittest.TestCase):
    def test_save_and_load_hdf5_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            self.obj_h5dm = HDF5DataManager()
            self.obj_h5dm.add_dataset('test_data', np.array([1, 2, 3]))
            self.obj_h5dm.save(tmpfile.name)
            loaded_manager = HDF5DataManager()
            loaded_manager.load(tmpfile.name)
            self.assertIn('test_data', loaded_manager.datasets)
            os.remove(tmpfile.name)
    
    def test_save_load_multiple_datasets(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            self.obj_h5dm = HDF5DataManager()
            self.obj_h5dm.add_dataset('temp', np.array([20, 21, 22]))
            self.obj_h5dm.add_dataset('pressure', np.array([101, 102, 103]))
            self.obj_h5dm.save(tmpfile.name)
            loaded_manager = HDF5DataManager()
            loaded_manager.load(tmpfile.name)
            self.assertIn('temp', loaded_manager.datasets)
            self.assertIn('pressure', loaded_manager.datasets)
            os.remove(tmpfile.name)
    
    def test_file_structure_validation(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            with h5py.File(tmpfile.name, 'w') as f:
                f.create_dataset('invalid_data', data=np.array([1, 2, 3]))
            self.obj_h5dm = HDF5DataManager()
            with self.assertRaises(KeyError):
                self.obj_h5dm.load(tmpfile.name)
            os.remove(tmpfile.name)
