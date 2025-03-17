import numpy as np
from Bifrost import data_manager

def test_add_axis():
    dm = data_manager.DataManager()
    dm.add_axis(data_manager.AxisType.TIME.value, 'time_vi_cadence', data_manager.AxisUnits.SECONDS.value)
    assert 'time_vi_cadence' in dm.axes
    assert dm.axes['time_vi_cadence']['type'] == data_manager.AxisType.TIME.value
    assert dm.axes['time_vi_cadence']['units'] == data_manager.AxisUnits.SECONDS.value
    print("test_add_axis passed.")

def test_add_dataset():
    dm = data_manager.DataManager()
    dm.add_axis(data_manager.AxisType.TIME.value, 'time_vi_cadence', data_manager.AxisUnits.SECONDS.value)
    data = np.array([1, 2, 3])
    axes_info = [{'name': 'time_vi_cadence', 'type': data_manager.AxisType.TIME.value, 'units': data_manager.AxisUnits.SECONDS.value}]
    dm.add_dataset('volume_integrated_values', 'mach_number', data, axes_info)
    assert 'volume_integrated_values' in dm.datasets
    assert 'mach_number' in dm.datasets['volume_integrated_values']
    assert np.array_equal(dm.datasets['volume_integrated_values']['mach_number']['data'], data)
    assert dm.datasets['volume_integrated_values']['mach_number']['axes'] == ['time_vi_cadence']
    print("test_add_dataset passed.")

def test_set_metadata():
    dm = data_manager.DataManager()
    dm.set_metadata('experiment_description', 'Example Experiment')
    assert dm.metadata['experiment_description'] == 'Example Experiment'
    print("test_set_metadata passed.")

def test_cull_unused_axes():
    dm = data_manager.DataManager()
    dm.add_axis(data_manager.AxisType.TIME.value, 'time_vi_cadence', data_manager.AxisUnits.SECONDS.value)
    dm.add_axis(data_manager.AxisType.BIN_EDGES.value, 'kappa_bin_edges', data_manager.AxisUnits.UNITLESS.value)
    dm.add_dataset('volume_integrated_values', 'mach_number', np.array([1, 2, 3]), [{'name': 'time_vi_cadence', 'type': data_manager.AxisType.TIME.value}])
    dm.cull_unused_axes()
    assert 'time_vi_cadence' in dm.axes
    assert 'kappa_bin_edges' not in dm.axes
    print("test_cull_unused_axes passed.")

def test_check_consistency():
    dm = data_manager.DataManager()
    dm.add_axis(data_manager.AxisType.TIME.value, 'time_vi_cadence', data_manager.AxisUnits.SECONDS.value)
    dm.add_axis(data_manager.AxisType.BIN_EDGES.value, 'kappa_bin_edges', data_manager.AxisUnits.UNITLESS.value)
    dm.add_dataset('volume_integrated_values', 'mach_number', np.array([1, 2, 3]), [{'name': 'time_vi_cadence', 'type': data_manager.AxisType.TIME.value}])
    dm.add_dataset('pdfs', 'kappa_pdf_p16', np.array([4, 5, 6]), [{'name': 'kappa_bin_edges', 'type': data_manager.AxisType.BIN_EDGES.value}])
    try:
        dm.check_consistency()
        print("test_check_consistency passed.")
    except ValueError as e:
        print(f"Consistency error: {e}")

def run_tests():
    test_add_axis()
    test_add_dataset()
    test_set_metadata()
    test_cull_unused_axes()
    test_check_consistency()

run_tests()
