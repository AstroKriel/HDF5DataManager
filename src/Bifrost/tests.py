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