## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from enum import Enum


## ###############################################################
## RELEVANT UNITS FOR YOUR WORK
## ###############################################################
class DatasetUnits(Enum):
  NOT_SPECIFIED = "not_specified"
  DIMENSIONLESS = "dimensionless"


## ###############################################################
## DATASET MANAGER
## ###############################################################
class DatasetObject:
  def __init__(
      self,
      group, name, values, list_axis_objs,
      units = DatasetUnits.NOT_SPECIFIED.value,
      notes = "",
    ):
    self.group  = group
    self.name   = name
    self.values = numpy.array(values)
    ## store units as a string
    units = self.cast_units_to_string(units)
    self.units = units
    self.notes = notes
    self.list_axis_objs = list_axis_objs

  @staticmethod
  def _validate_inputs(group, name, values, units, notes):
    if not isinstance(group, str) or (group.strip() == ""): raise ValueError("Dataset group must be a non-empty string.")
    if not isinstance(name, str) or (name.strip() == ""): raise ValueError("Dataset name must be a non-empty string.")
    if not isinstance(values, (list, numpy.ndarray)): raise TypeError("Dataset values must be a list or numpy array.")
    values = numpy.array(values)
    if values.size == 0: raise ValueError("Dataset values cannot be empty.")
    if not numpy.issubdtype(values.dtype, numpy.number): raise TypeError("Dataset values must be numeric (either integers or floats).")
    if not isinstance(units, DatasetUnits): raise TypeError(f"Invalid dataset unit: {units}. Must be an instance of DatasetUnits.")
    if not isinstance(notes, str): raise TypeError("Notes must be a string.")

  @staticmethod
  def cast_units_to_string(units):
    if not isinstance(units, (str, DatasetUnits)): raise TypeError("Error: Provided `units` was neither a string or an element of DatasetUnits.")
    if isinstance(units, DatasetUnits): units = units.value
    return units

  @staticmethod
  def create_dict_inputs(
      group, name, values,
      units = DatasetUnits.NOT_SPECIFIED,
      notes = "",
    ):
    DatasetObject._validate_inputs(group, name, values, units, notes)
    units = DatasetObject.cast_units_to_string(units)
    return {
      "group"  : group,
      "name"   : name,
      "values" : numpy.array(values),
      "units"  : units,
      "notes"  : notes,
    }

  def get_dict(self):
    return {
      "group"  : self.group,
      "name"   : self.name,
      "values" : self.values,
      "units"  : self.units,
      "notes"  : self.notes,
      "list_axis_dicts": [
        obj_axis.get_dict()
        for obj_axis in self.list_axis_objs
      ],
    }

  def add(self, dataset_values_in, list_axis_values_in):
    self.reindex(
      dataset_values_in   = dataset_values_in,
      list_axis_values_in = list_axis_values_in,
    )

  def reindex(self, list_axis_values_in, dataset_values_in=None):
    num_axes_in = len(list_axis_values_in)
    num_dataset_dim = len(self.values.shape)
    if num_dataset_dim != num_axes_in:
      raise ValueError(f"Error: The number of provided axis do not match the dimensions of the existing dataset (`{num_axes_in}` != `{num_dataset_dim}`).")
    ## collect the stored axis values before updating
    list_axis_values_old = [
      numpy.array(obj_axis_old.values, copy=True)
      for obj_axis_old in self.list_axis_objs # before axes have been updated
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
    ## if no `in` data was provided, then the new space will be filled with `nan`s
    ## note: `in` may not necessarily be `new`
    ## where `old` == `in` -> overwrite
    ## where `old` != `in` -> merge
    updated_dataset_shape = tuple(
      len(obj_axis_updated.values)
      for obj_axis_updated in self.list_axis_objs # axes have been updated with inputs
    )
    dataset_values_updated = numpy.full(updated_dataset_shape, numpy.nan)
    dataset_indices_old = tuple(
      numpy.searchsorted(obj_axis_updated.values, old_axis_values)
      for obj_axis_updated, old_axis_values in zip(
        self.list_axis_objs,
        list_axis_values_old
      )
    )
    dataset_values_updated[numpy.ix_(*dataset_indices_old)] = self.values
    if dataset_values_in is not None:
      dataset_indices_in = tuple(
        numpy.searchsorted(obj_axis_updated.values, new_axis_values)
        for obj_axis_updated, new_axis_values in zip(
          self.list_axis_objs,
          list_axis_values_in
        )
      )
      dataset_values_updated[numpy.ix_(*dataset_indices_in)] = dataset_values_in
    self.values = dataset_values_updated


## END OF MODULE